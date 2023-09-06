from main import INITIAL_INVESTMENT


class Trade:
    initial_investment = INITIAL_INVESTMENT

    def __init__(self, stock1, stock2):
        self.stock1 = stock1
        self.stock2 = stock2
        self.invest = {stock1: 0, stock2: 0}
        self.stock_price = {stock1: None, stock2: None}

    def long(self, stock, buy_amount):
        self.invest[stock] += buy_amount

    def short(self, stock, sell_amount):
        self.invest[stock] -= sell_amount
    
    def update_stock_price(self, stock1_price, stock2_price):
        self.stock_price[self.stock1] = stock1_price
        self.stock_price[self.stock2] = stock2_price

