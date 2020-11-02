import cash_db

class_names = [
    'blackbean', 'herbsalt', 'homerun', 'lion', 'narangd', 'rice', 'sixopening', 'skippy', 'BlackCap', 'CanBeer', 'doritos',
    'Glasses', 'lighter', 'mountaindew', 'pepsi', 'Spoon',  'tobacco', 'WhiteCap', 'note'
]

#class의 길이를 정한다.
num_classes = len(class_names)

class_prices = [
    1000, 800, 1500, 6000, 1000, 1500, 800, 800, 25000, 2000, 1500, 50000, 4000, 1000, 1000, 1000, 1500, 30000, 2000
]

class_counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

mydb = cash_db.MySql(user = 'root', password = 'root', db_name = 'smart_cashier')
mydb.connect()

for idx in range(0, len(class_names)):
    res = mydb.check_exist_data(class_names[idx])
    print(class_names[idx], res)
    if res >= 1:
        continue
    mydb.insert_data(class_names[idx], class_prices[idx], class_counts[idx])
mydb.close_db()
