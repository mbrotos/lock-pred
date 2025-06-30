import re
import os
import csv
os.add_dll_directory("C:\\Program Files\\IBM\\SQLLIB\\bin")
import ibm_db
import random
import time
 

# Database connection details
db_name = 'testdb3'
db_user = 'db2admin'
db_password = 'root'
db_host = 'localhost'
db_port = '25000'



# Initialize DB2 connection
conn_str = f"DATABASE={db_name};HOSTNAME={db_host};PORT={db_port};PROTOCOL=TCPIP;UID={db_user};PWD={db_password};"

try:
    print("Trying to connect.")
    conn = ibm_db.connect(conn_str, "", "")
    print("Connected to the database successfully.")
except Exception as e:
    print(f"Failed to connect to the database: {str(e)}")
    exit()
 
  
# Function to insert data into ORDER_LINE table
def insert_order_line(conn, order_id, order_number):
    sql = """INSERT INTO ORDER_LINE (
                OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER, OL_I_ID, OL_SUPPLY_W_ID, OL_DELIVERY_D, OL_QUANTITY, OL_AMOUNT, OL_DIST_INFO
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    
    # Random values for testing
    ol_w_id = random.randint(1, 10)            # Warehouse ID between 1 and 5
    ol_d_id = random.randint(1, 100)           # District ID between 1 and 10
    ol_i_id = random.randint(1, 100000)         # Item ID between 1 and 1000
    ol_supply_w_id = ol_w_id                  # Supply warehouse is the same as ordering warehouse
    ol_quantity = random.randint(1, 10)       # Quantity ordered between 1 and 10
    ol_amount = round(random.uniform(1.0, 100.0), 2)  # Random amount between $1 and $100
    ol_dist_info = f"Dist {order_number}"     # Distribution info
    
    # Execute the query
    stmt = ibm_db.prepare(conn, sql)
    ibm_db.bind_param(stmt, 1, ol_w_id)
    ibm_db.bind_param(stmt, 2, ol_d_id)
    ibm_db.bind_param(stmt, 3, order_id)
    ibm_db.bind_param(stmt, 4, order_number)
    ibm_db.bind_param(stmt, 5, ol_i_id)
    ibm_db.bind_param(stmt, 6, ol_supply_w_id)
    ibm_db.bind_param(stmt, 7, None)          # Delivery date is NULL initially
    ibm_db.bind_param(stmt, 8, ol_quantity)
    ibm_db.bind_param(stmt, 9, ol_amount)
    ibm_db.bind_param(stmt, 10, ol_dist_info)
    
    try:
        ibm_db.execute(stmt)
        print(f"Inserted order line for order {order_id}, line number {order_number}")
    except Exception as e:
        print(f"Error inserting order line: {e}")

# Main function
def main():
    # Connect to the database
    try:
        conn = ibm_db.connect(conn_str, "", "")
        print("Connected to the database")
        
        for i in range(1, 100000):  # Loop 100 times
            order_id = 1000 + i  # Generating unique Order ID
            order_number = i      # Order line number
            
            # Insert the order line
            insert_order_line(conn, order_id, order_number)
            
            time.sleep(1)  # Delay between inserts for demonstration purposes (optional)
    
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
    
    finally:
        # Close the connection
        if conn:
            ibm_db.close(conn)
            print("Connection closed")

if __name__ == "__main__":
    main()
