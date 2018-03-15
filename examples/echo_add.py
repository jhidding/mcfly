from noodles import schedule


@schedule
def echo_add(x, y):
    print('transmogrifying', x, 'and', y)
    return x + y
