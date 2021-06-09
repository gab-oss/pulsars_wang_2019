import re
import xml.etree.cElementTree as ET
import numpy
import random

################################################################################

def readDataBlock(xmlnode):
    """ Turn any 'DataBlock' XML node into a numpy array of floats
    """
    vmin = float(xmlnode.get('min'))
    vmax = float(xmlnode.get('max'))
    string = xmlnode.text
    string = re.sub("[\t\s\n]", "", string)
    data = numpy.asarray(
        bytearray.fromhex(string),
        dtype = float
        )
    return data * (vmax - vmin) / 255. + vmin
    

class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path.
        """
        xmlroot = ET.parse(fname).getroot()
        
        # Read Coordinates
        coordNode = xmlroot.find('head').find('Coordinate')
        self.rajd = float(coordNode.find('RA').text)
        self.decjd = float(coordNode.find('Dec').text)
        
        # Separate PDMP & FFT sections
        for section in xmlroot.findall('Section'):
            if 'pdmp' in section.get('name').lower():
                opt_section = section
            else:
                fft_section = section
            
        # Best values as returned by PDMP
        opt_values = {
            node.tag : float(node.text)
            for node in opt_section.find('BestValues').getchildren()
            }
        
        self.bary_period = opt_values['BaryPeriod']
        self.topo_period = opt_values['TopoPeriod']
        self.dm = opt_values['Dm']
	self.snr = opt_values['Snr']
	self.width = opt_values['Width']
        
        ##### P-DM plane #####
        pdmNode = opt_section.find('SnrBlock')
        # DmIndex
        string = pdmNode.find('DmIndex').text
        dm_index = numpy.asarray(map(float, string.split()))
        
        # PeriodIndex
        string = pdmNode.find('PeriodIndex').text
        period_index = numpy.asarray(map(float, string.split()))
        period_index /= 1.0e12 # Picoseconds to seconds

        # S/N data
        pdmDataNode = pdmNode.find('DataBlock')
        pdm_plane = readDataBlock(pdmDataNode).reshape(
            dm_index.size, 
            period_index.size
            )
        
        # Pack all P-DM plane arrays into a tuple
        self.pdm_plane = (period_index, dm_index, pdm_plane)
        
        ### Sub-Integrations
        subintsNode = opt_section.find('SubIntegrations')
        nsubs = int(subintsNode.get('nSub'))
        nbins = int(subintsNode.get('nBins'))
        self.subints = readDataBlock(subintsNode).reshape(nsubs, nbins)
        
        ### Sub-Bands
        subbandsNode = opt_section.find('SubBands')
        nsubs = int(subbandsNode.get('nSub'))
        nbins = int(subbandsNode.get('nBins'))
        self.subbands = readDataBlock(subbandsNode).reshape(nsubs, nbins)
        
        ### Profile
        profileNode = opt_section.find('Profile')
        self.profile = readDataBlock(profileNode)
        
        ##### Parse FFT Section (PEASOUP Data) #####
        fft_values = {
            node.tag : float(node.text)
            for node in fft_section.find('BestValues').getchildren()
            }
        self.accn = fft_values['Accn']
        self.hits = fft_values['Hits']
        self.rank = fft_values['Rank']
        self.fftsnr = fft_values['SpectralSnr']
        
        ### DmCurve: FFT S/N vs. PEASOUP Trial DM, at best candidate acceleration
        dmcurve_node = fft_section.find('DmCurve')
        
        text = dmcurve_node.find('DmValues').text
        dm_values = numpy.asarray(map(float, text.split()))
        text = dmcurve_node.find('SnrValues').text
        snr_values = numpy.asarray(map(float, text.split()))
        
        # Pack the DM curve into a tuple of arrays
        self.dm_curve = (dm_values, snr_values)
        
        ### AccnCurve: FFT S/N vs. PEASOUP Trial Acc, at best candidate DM
        accncurve_node = fft_section.find('AccnCurve')
        
        text = accncurve_node.find('AccnValues').text
        accn_values = numpy.asarray(map(float, text.split()))
        text = accncurve_node.find('SnrValues').text
        snr_values = numpy.asarray(map(float, text.split()))
        
        # Pack the Accn curve into a tuple of arrays
        self.accn_curve = (accn_values, snr_values)

################################################################################

def phase_plots(cand):
    import pylab as plt
    plt.figure(1, figsize=(9, 7), dpi=70)
    plt.subplot(311)
    plt.imshow(cand.subbands, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Bands')
    plt.ylabel('Band Index')
    
    plt.subplot(312)
    plt.imshow(cand.subints, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Integrations')
    plt.ylabel('Integration Index')
    
    plt.subplot(313)
    plt.bar(xrange(cand.profile.size), cand.profile, width=1)
    plt.xlim(0, cand.profile.size)
    plt.xlabel('Phase Bin Index')
    
    plt.tight_layout()
    # plt.show()
    

def bullseye_plot(cand):
    import pylab as plt
    p, dm, snr = cand.pdm_plane
    
    plt.figure(2, figsize=(7, 5), dpi=80)
    # IMPORTANT NOTE: imshow() must be called with origin='lower' here, otherwise 
    # the DM values on the Y axis are reversed (and therefore wrong).
    plt.imshow(
        snr, 
        extent=[p.min(), p.max(), dm.min(), dm.max()], 
        aspect='auto', 
        origin='lower',
        interpolation='nearest'
        )
    plt.xlabel('Period Correction (s)')
    plt.ylabel('Trial DM')
    
    cb = plt.colorbar()
    cb.set_label('Folded S/N')
    
    plt.tight_layout()
   # plt.show()


def find_peak(cand):
    max_p = cand.profile[0]
    max_index = 0
    i = 0
    for p in cand.profile[1:]:
        i += 1
        if p > max_p:
            max_p = p
            max_index = i
    return max_index


def find_peak_subbands(cand):
    max_p = cand.subbands[0]
    max_index = 0
    i = 0
    for p in cand.subbands[1:]:
        i += 1
        if p > max_p:
            max_p = p
            max_index = i
    return max_index


def transform_subbands(cand):
    peak_index = find_peak(cand)
    middle_index = (len(cand.subbands[0]) / 2) - 1
    diff = peak_index - middle_index
    new_subbands = numpy.zeros((19, 64))
    i = 0
    for arr in cand.subbands:
        new_arr = arr[diff:]
        new_array = numpy.append(new_arr, arr[:diff])
        new_subbands[i] = new_array
        i += 1
    cand.subbands = new_subbands
    return cand


def transform_subints(cand):
    peak_index = find_peak(cand)
    middle_index = (len(cand.subints[0]) / 2) - 1
    diff = peak_index - middle_index
    new_subints = numpy.zeros((19, 64))
    i = 0
    for arr in cand.subints:
        new_arr = arr[diff:]
        new_array = numpy.append(new_arr, arr[:diff])
        new_subints[i] = new_array
        i += 1
    cand.subints = new_subints
    return cand


def random_shift(subs):
    diff = random.randint(0, len(subs[0]))
    new_subs = numpy.zeros((19, 64))
    i = 0
    for arr in subs:
        new_arr = arr[diff:]
        new_array = numpy.append(new_arr, arr[:diff])
        new_subs[i] = new_array
        i += 1
    return new_subs    


def subbands_plot(subbands, name):
    import pylab as plt
    plt.figure(1, figsize=(1, 1), dpi=64)
    plt.imshow(subbands, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    #plt.title('Sub-Bands')
    #plt.ylabel('Band Index')
    #plt.tight_layout()
    plt.axis('off')
    plt.savefig(name, dpi=64)
    

def subints_plot(subints, name):
    import pylab as plt
    plt.figure(1, figsize=(1, 1), dpi=64)
    plt.imshow(subints, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    #plt.title('Sub-Integrations')
    #plt.ylabel('Integration Index')
    #plt.tight_layout()
    plt.axis('off')
    plt.savefig(name, dpi=64)

    
def save_subplots(path, dirpath, ext, is_pulsar):
    import os

    bands_str = "_subbands"
    ints_str = "_subints"
    cand = Candidate(path)
    directory, fname = os.path.split(
        os.path.abspath(path)
        )
    fname = fname.split('.')[0]

    subbands_plot(cand.subbands, os.path.join(dirpath, fname + bands_str + ext))
    subints_plot(cand.subints, os.path.join(dirpath, fname + ints_str + ext))

    if is_pulsar:
        transform_subbands(cand)
        transform_subints(cand)
    subbands_plot(cand.subbands, os.path.join('./trans_pulsar_plots/', fname + bands_str + ext))
    subints_plot(cand.subints, os.path.join('./trans_pulsar_plots/', fname + ints_str + ext))


def add_plots(plots, weights):
    sum = plots[0] * weights[0]
    for i in xrange(1, len(plots)):
        weighted_plot = plots[i] * weights[i]
        sum = numpy.add(sum, weighted_plot)
    return sum


def make_artificial_pulsar():
    ext = ".png"
    source_dir = './pulsars/'
    target_dir = './artificial_plots/'
    fname = 'art_pulsar_'
    bands_str = 'subbands'
    ints_str = 'subints'

    pulsar_files = os.listdir(source_dir)
    src_pulsars = []

    weights_dividing_points = []
    for _ in xrange(2):
        weights_dividing_points.append(random.uniform())
    weights_dividing_points.sort()
    a = weights_dividing_points[0]
    b = weights_dividing_points[1] - weights_dividing_points[0]
    c = 1 - weights_dividing_points[1]

    for i in xrange(3):
        src_pulsar_file = random.choice(pulsar_files)
        #save_subplots(source_dir + src_pulsar_file, target_dir, ext, True)
        fname = fname + src_pulsar_file[-9:-5] + '_'

        src_pulsar = Candidate(source_dir + src_pulsar_file)
        transform_subbands(src_pulsar)
        transform_subints(src_pulsar)

        #phase_plots(src_pulsar)
        src_pulsars.append(src_pulsar)

    src_subbands = []
    src_subints = []

    for pulsar in src_pulsars:
        src_subbands.append(pulsar.subbands)
        src_subints.append(pulsar.subints)

    subbands = add_plots(src_subbands, [a, b, c])
    subints = add_plots(src_subints, [a, b, c])

    shifted_subbands = []
    shifted_subints = []
    for band in subbands:
        shifted_subbands.append(random_shift(band))
    for band in subints:
        shifted_subints.append(random_shift(band))

    subbands_plot(shifted_subbands, os.path.join(target_dir, fname + bands_str + ext))
    subints_plot(shifted_subints, os.path.join(target_dir, fname + ints_str + ext))


def prepare_data():
    positive_examples = './pulsars/'
    negative_examples = './RFI/'
    positive_plots = './pulsar_plots/'
    negative_plots = './RFI_plots/'

    for filename in os.listdir(positive_dirs):
        save_subplots(positive_examples + filename, positive_plots, '.png', True)

    for filename in os.listdir(negative_dirs):
        save_subplots(negative_examples + filename, negative_plots, '.png', False)  

    for _ in range(0, 20793):
        make_artificial_pulsar()



################################################################################

if __name__ == '__main__':
    import os

    # # Load example.phcx file (must be in the same directory as this python script)
    # directory, fname = os.path.split(
    #     os.path.abspath(__file__)
    #     )
    # cand = Candidate(
    #     os.path.join(directory, './pulsars/pulsar_0023.phcx')
    #     )
    
    # Make some cool plots
    
    # znajdz w cand.profile najwieksza wartosc i jej indeks to indeks peaka, potem przesun w subbands i subints w kazdej tablicy ten indeks na srodek
    # rozdziel funkcje phase_plots na 2
    # zrob funkcje ktora zapisuje do plikow obrazy dla sciezki
    # zrob taka funkcje dla listy folderow
    # funkcja ktora dodaje 3 dwuwyamirowe tablice z wagami alfa, beta, gamma

    # rozpakuj pozytywne, przetworz, zrob obrazy
    # dorob nowe pulsary (wez trzy pliki, przemiel, zapisz dwa obrazy - oddzielny folder na sztuczne pulsary)
    # rozpakuj negatywne i zrob obrazy
    # polacz foldery i wsadz w tensorflow
    # w razie problemow z pamiecia uzyj po kawalku z kazdego folderu

    #phase_plots(cand)
    # bullseye_plot(cand)
    
    # transform_subbands(cand)
    # transform_subints(cand)
    # phase_plots(cand)
    # subbands_plot(cand.subbands, "subbands.png")
    # subints_plot(cand.subints, "subints.png")


    # cand0 = Candidate('./pulsars/pulsar_0000.phcx')
    # cand1 = Candidate('./pulsars/pulsar_0001.phcx')
    # cand2 = Candidate('./pulsars/pulsar_0002.phcx')
    # a = 0.333
    # b = 0.333
    # c = 1 - (a + b)
    # add_plots([cand0.subbands, cand1.subbands, cand2.subbands], [a, b, c])

    # make_artificial_pulsar()
    prepare_data()
