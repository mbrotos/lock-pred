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

def get_valid_combinations(conn):
    sql = "SELECT OL_O_ID, OL_W_ID, OL_D_ID, OL_NUMBER FROM ORDER_LINE"
    try:
        stmt = ibm_db.exec_immediate(conn, sql)
        rows = []
        result = ibm_db.fetch_assoc(stmt)
        while result:
            rows.append((result['OL_O_ID'], result['OL_W_ID'], result['OL_D_ID'], result['OL_NUMBER']))
            result = ibm_db.fetch_assoc(stmt)
        print(f"Retrieved {len(rows)} valid combinations from the database.")
        return rows
    except Exception as e:
        print(f"Error retrieving valid combinations: {e}")
        return []
  
# Function to update data in the ORDER_LINE table
def update_order_line(conn, combination, order_number):
    sql = """UPDATE ORDER_LINE
             SET  OL_SUPPLY_W_ID = ?, OL_DELIVERY_D = ?, OL_QUANTITY = ?, OL_AMOUNT = ?, OL_DIST_INFO = ?
             WHERE OL_O_ID = ? AND OL_W_ID = ? AND OL_D_ID = ? AND  OL_NUMBER = ? """
     
    # Extract valid identifiers
    OL_O_ID, ol_w_id, ol_d_id, OL_NUMBER = combination

    # Generate random values for the update
    ol_supply_w_id = ol_w_id                  # Supply warehouse is the same as ordering warehouse
    ol_quantity = random.randint(1, 10)       # Quantity ordered between 1 and 10
    ol_amount = round(random.uniform(1.0, 100.0), 2)  # Random amount between $1 and $100
    ol_dist_info = f"Dist {order_number}"     # Distribution info
    ol_delivery_d = None  # Delivery date can be NULL for testing

    # Execute the query
    stmt = ibm_db.prepare(conn, sql)

    ibm_db.bind_param(stmt, 1, ol_supply_w_id)
    ibm_db.bind_param(stmt, 2, ol_delivery_d)
    ibm_db.bind_param(stmt, 3, ol_quantity)
    ibm_db.bind_param(stmt, 4, ol_amount)
    ibm_db.bind_param(stmt, 5, ol_dist_info)
    ibm_db.bind_param(stmt, 6, OL_O_ID)
    ibm_db.bind_param(stmt, 7, ol_w_id)
    ibm_db.bind_param(stmt, 8, ol_d_id)
    ibm_db.bind_param(stmt, 9, OL_NUMBER)
 
    
    try:
        ibm_db.execute(stmt)
        ibm_db.commit(conn)  # Force commit after each update
        print(f"Updated order line for OL_O_ID {OL_O_ID}, OL_W_ID {ol_w_id}, OL_D_ID {ol_d_id}, OL_NUMBER {OL_NUMBER}")
    except Exception as e:
        print(f"Error updating order line: {e}")

# Main function
def main():
    # Connect to the database
    try:
        conn = ibm_db.connect(conn_str, "", "")
        print("Connected to the database")

        # Retrieve valid combinations
        valid_combinations = get_valid_combinations(conn)
        
        if not valid_combinations:
            print("No valid combinations found. Exiting.")
            return
        

        update_count = 0  # Counter to keep track of updates
        ibm_db.exec_immediate(conn, "SET CURRENT ISOLATION CS")

        for i in range(1, 101002):  # Loop 100 times
            order_id = 1000 + i  # Generating unique Order ID
            order_number = i      # Order line number
                        # Select a random combination
            combination = random.choice(valid_combinations)
            
            # Update the order line
            update_order_line(conn, combination, order_number)
     
            # Increment the counter and print the update count
            update_count += 1
            print(f"Total updates executed so far: {update_count}")

            time.sleep(1)  # Delay between updates for demonstration purposes (optional)
    
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
    
    finally:
        # Close the connection
        if conn:
            ibm_db.close(conn)
            print("Connection closed")

if __name__ == "__main__":
    main()
