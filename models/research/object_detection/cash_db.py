import pymysql

class MySql:
    #db를 연결하기 위한 모든 정보를 얻는다.
    def __init__(self, user, password, db_name, host = '127.0.0.1', port = 3306):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
    
    #db에 연결을 시도한다.
    def connect(self):
        self.conn = pymysql.connect(host = self.host, port = self.port, user = self.user, password = self.password,  \
            db = self.db_name, charset = 'utf8')
        self.cursor = self.conn.cursor()
        print('DB CONNECTED COMPLETE!!!!')
    
    #테이블을 만든다.
    def create_table(self, table_name):

        sql = "create table " + table_name + "(product_name varchar(100) NOT NULL, price INT UNSIGNED NOT NULL, product_count INT NOT NULL, PRIMARY KEY(product_name), \
        update_time TIMESTAMP NOT NULL default NOW());"
        self.table_name = table_name
        self.cursor.execute(sql)
        print('CREATE TABLE COMPLETE!!!!')

    def insert_data(self, product_name, price, product_count):
        sql = '''
            INSERT INTO cashier (product_name, price, product_count) VALUES (%s, %s, %s);
            '''
        
        self.cursor.execute(sql, (product_name, price, product_count))
        self.conn.commit()
    
    def delete_data(self, product_name):
        if self.table_name == None:
            print('There is not Table')
            return

        sql = "DELETE FROM " + self.table_name
        sql += " WHERE product_name = %s"

        self.cursor.execute(sql, (self.table_name, "product_name", product_name))
    
    #마지막으로 나온 결과 값을 db에 반영시켜준다.
    def apply_result_db(self, product_name, product_count):
        if self.table_name == None:
            print('There is not Table')
            return

        sql = "UPDATE " + self.table_name
        sql += " SET product_count = %s, update_time = NOW()"
        sql += " WHERE product_name = %s"

        self.cursor.execute(sql, (product_count, product_name))
        self.conn.commit() 

    #여러가지를 확인해주기 위한 코드
    def show_table(self):
        self.cursor.execute("SHOW TABLES")
    
    #table 존재하는지 확인
    def check_exist_table(self, table_name):
        sql = "SHOW TABLES LIKE %s"
        self.cursor.execute(sql, table_name)
        #튜플로서 결과를 return한다.
        result = self.cursor.fetchall()
        result = list(result)
        print(len(result))
        if len(result) >= 1:
            print('DB ALREADY EXISTED, you can insert, delete, etc.. using class function')
            self.table_name = table_name
            return len(result)
        else:
            print('DB NOT EXISTED, CREATE TABLE FIRST USING create_table(table_name) functions')
            return len(result)
    
    #상품명이 같은 데이터가 존재하면, insert를 하지 않는다.
    def check_exist_data(self, product_name):
        if self.table_name == None:
            self.table_name = "cashier"
        sql = "SELECT * FROM " + self.table_name
        sql += " WHERE product_name = %s"

        self.cursor.execute(sql, product_name)
        result = self.cursor.fetchall()
        return len(result)
    
    def select_all(self):
        if self.table_name == None:
            self.table_name = 'cashier'
        sql = "SELECT * FROM "
        sql += self.table_name

        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def select_price(self, product_name):
        sql = "SELECT * FROM " + self.table_name
        sql += " WHERE product_name = %s"

        self.cursor.execute(sql, (product_name))
        result = self.cursor.fetchall()
        #가격을 돌려준다.
        if len(result) <= 0:
            return len(result)
        
        price = [item[1] for idx, item in enumerate(result) if idx == 0]
        return price[0]

    def close_db(self):
        print('CLOSE THE CURSOR')
        self.cursor.close()
        
    


    