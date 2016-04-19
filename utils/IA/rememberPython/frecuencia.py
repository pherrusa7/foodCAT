import unicodedata


def frecuencia(location):
    '''
    This function takes a text file as input and count how many times each Catalan character appears in the text
    '''

    catalanDictonary = {'a':0, 'b':0, 'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,
                        'o':0,'p':0,'q':0,'r':0,'s':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0}
    
    #Open and read the text as String
    text = openFile(location)
    
    if text:        
    
        # Convert uppercase to lowercase
        text = text.lower()
        
        # take out any diacritical mark
        text = remove_diacritic(unicode(text,'ISO-8859-1'))

        #Count characers
        for char in text:
            if catalanDictonary.has_key(char):
                catalanDictonary[char] += 1
       

        print catalanDictonary
    

    
    return

def openFile(location):
    '''
    This function opens a file and returns a string with the contents.
    Returns 0 if unsuccessful
    '''

    try:
        f = open(location)
        s = f.read()
        return s
    except IOError :
        print "I/O error: File doesn't appear to exist."
    except:
        print "Unexpected error:"
    return 0

def remove_diacritic(input):
    '''
    Accept a unicode string and return a normal string without any diacritical marks.
    '''

    return unicodedata.normalize('NFKD',input).encode('ASCII','ignore')


