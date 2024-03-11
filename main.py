import random
import csv

TABLE_SIZE = 10
global cities
cities = []

def initCities():
    global cities
    cities = []
    
    for i in range(0, TABLE_SIZE):
        cities.insert(i, [])
        for j in range(0, TABLE_SIZE):
            if random.randint(0, 50) == 0:
            # if i == 5 and j == 5:
                cities[i].insert(j, 'ville')
            else:
                cities[i].insert(j, 'rien')
                
    # for i in range(0, TABLE_SIZE):
    #     for j in range(0, TABLE_SIZE):
            
    #         if cities[i][j] == 'ville':     
                
    #             for x in range(i+1, i+4, TABLE_SIZE):
    #                 for y in range(j+1, j+4, TABLE_SIZE):
                        
    #                     if x < len(cities) and y < len(cities[0]):                    
    #                         if cities[x][y] == 'ville':
    #                             print('ville base: x ', str(i) + " - y: ", str(j))
    #                             print('ville voisine: x ', str(x) + " - y: ", str(y) + '\n')
                            
    #                         if cities[x][y] == 'ville':
    #                             for xx in range(i+1, x):
    #                                 if cities[xx][y] != 'ville':
    #                                     cities[xx][y] = 'chemin'
    #                             for yy in range(j+1, y):
    #                                 if cities[x][yy] != 'ville':
    #                                     cities[x][yy] = 'chemin' 
    
    for i in range(0, TABLE_SIZE):
        for j in range(0, TABLE_SIZE):
            if cities[i][j] == 'ville':
                for x in range(max([0, i-3]), min([i+4, TABLE_SIZE])):
                    for y in range(max([0, j-3]), min([j+4, TABLE_SIZE])):
                        if i != x or j != y:
                            if cities[x][y] == 'ville':
                                print('city 1: ', str(i), str(j)) 
                                print('city 2: ', str(x), str(y), "\n") 
                                for xx in range(min([i, x]), max([i, x])):
                                    if cities[xx][y] != 'ville':
                                        cities[xx][y] = 'chemin'
                                for yy in range(min([j, y]), max([j, y])):
                                    if cities[x][yy] != 'ville':
                                        cities[x][yy] = 'chemin' 
                            # else:
                            #     cities[x][y] = 'chemin'
                
def display():
    global cities
    
    buffer = '  '
    for i in range(0, TABLE_SIZE):
        buffer = buffer + str(i) + ' '
        
    print(buffer)
    
    for i in range(0, TABLE_SIZE):
        buffer = ''
        for j in range(0, TABLE_SIZE):
            if cities[i][j] == 'ville':
                buffer = buffer + '◼'
            elif cities[i][j] == 'chemin':
                buffer = buffer + '⧄'
            else:
                buffer = buffer + '_'
            buffer = buffer + ' '
        
        print(str(i) + ' ' + buffer)
    print('\n◼ ville')
    print('_ vide')

initCities()
display()