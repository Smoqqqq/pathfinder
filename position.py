class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self) -> str:
        return f'x: {self.x}, y: {self.y}'
    
    def __lt__(self, other):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))