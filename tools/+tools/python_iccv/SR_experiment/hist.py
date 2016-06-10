import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


def hist(img_path):
    im = misc.imread(img_path)
    print '/'.join(img_path.split('/')[-3:])
    print 'mean: ', im.mean()
    print 'max: ', im.max(), ',', 'min: ', im.mean()
    hist, bins = np.histogram(im, bins=256)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

# original, sr = hist.main()
def main():
    sr_path='data_SR_resized_to_256/apple_pie/80734.jpg'
    original_path='data_original_resized_to_256/apple_pie/80734.jpg'

    original = hist(sr_path)

    sr = hist(original_path)

    return original, sr

if __name__=='__main__':
    main()
