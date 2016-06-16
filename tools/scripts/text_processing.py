import os
import numpy as np
#import pdb; pdb.set_trace()

def main(fileName):
    txt = np.array(np.loadtxt(fileName, str, delimiter='\t'))
    new_txt = []

    for l in txt:
        # get the image name and add .jpg at the end
        newLine = '{}.jpg'.format( l.split(' ')[0].split('.')[0] )
        # add the class after a space
        newLine = newLine+' '+l.split(' ')[1]
        new_txt.append(newLine)

    # save as a txt
    new_fileName = fileName.split('.')[0]+'_new.'+fileName.split('.')[1]
    np.savetxt(new_fileName, np.array(new_txt), delimiter='\t', fmt="%s")

if __name__=='__main__':
    main('test.txt')
    main('train.txt')
    main('val.txt')
    main('test_just_foodCAT.txt')
