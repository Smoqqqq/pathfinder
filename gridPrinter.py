from bgColor import BgColor

class GridPrinter:
    def __init__(self) -> None:
        self.paddSize = 5
    
    def display(self, grid: dict) -> None:
        buffer = ''
        
        if len(grid) > 50:
            self.paddSize = 3
            
        if len(grid) > 100:
            self.paddSize = 1
        
        if len(grid) < 100:
            buffer = '\n' + ''.ljust(self.paddSize, ' ')
            for i in range(0, len(grid)):
                buffer = buffer + str(i).ljust(self.paddSize, ' ')
            
        print(buffer)
        
        for i in range(0, len(grid)):
            buffer = ''
            for j in range(0, len(grid)):
                if grid[i][j] == 'city':
                    buffer = buffer + str('◼').ljust(self.paddSize, ' ')
                elif grid[i][j] == 'highlight_city':
                    buffer = buffer + BgColor.GREEN + str('◼').ljust(self.paddSize, ' ') + BgColor.END_COLOR
                elif grid[i][j] == 'void':
                    buffer = buffer + str('.').ljust(self.paddSize, ' ')
                else:
                    buffer = buffer + BgColor.GREEN + str(grid[i][j]).ljust(self.paddSize, ' ') + BgColor.END_COLOR
            
            if len(grid) < 100:
                buffer = buffer + '\n'
            
            print(str(i).ljust(self.paddSize, ' ') + buffer)
                
        print('\n◼ ville')
        print('_ vide\n')