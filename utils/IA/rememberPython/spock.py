from random import randint

def spok():
    '''
    This function it's the game called: Pedra, Paper, Estisores, Spock, Llangardaix.
    Need to interact with the user. 
    '''
    
    key_win = {"Pedra": ["Llangardaix", "Estisores"], "Paper": ["Pedra","Spock"], "Estisores": ["Paper","Llangardaix"], "Spock": ["Pedra","Estisores"],"Llangardaix": ["Paper", "Spock"] }
    key_value = {1:"Pedra",2:"Paper",3:"Estisores",4:"Spock",5:"Llangardaix"}
    
    play = True
    badOption = " Entra una opció vàlida. "

    #Introduce the game
    print " Benvingut al Pedra, Paper, Estisores, Spock, Llangardaix "
    
    while(play):
        print "\n Juguem! \n"

        #Check correct user option
        key = input(" Escriu 1 per escollir Pedra, 2 per Paper, 3 per Estisores, 4 per Spock i 5 per Llangardaix :  ")     
        while key<1 or key>5 :
            print badOption
            key = input()
        value = key_value[key]
        print " Has tret " + value
            
        #See who win
        random = randint(0,4)
        computerGame = key_win.items()[random][0]
        print " L'ordinador a tret " + computerGame 
        if computerGame == value :
            print " Empat! "
        elif(computerGame in key_win[value]):
            print " Guanyes! "
        else:
            print " Perds! "

        #Check End game
        nextGame = raw_input(" Vols continuar jugant? s=si  n=no :")
        while not(nextGame=='s' or nextGame=='n'):
            print badOption
            nextGame = raw_input(" ")
        if nextGame == 'n':
             play = False

    print " Fins aviat!"
        
        

    return
        
    
