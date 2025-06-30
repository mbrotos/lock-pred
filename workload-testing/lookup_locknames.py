import re
import os
import csv
os.add_dll_directory("C:\\Program Files\\IBM\\SQLLIB\\bin")
import ibm_db

# Database connection details
db_name = 'testdb3'
db_user = 'db2admin'
db_password = 'root'
db_host = 'localhost'
db_port = '25000'
 
# Input and output files (adjust paths as needed)
INPUT_FILE = r"C:\TMU\postdoc-TMU\HammerDB\trace_dumps\ColumnsTrace.flw"
OUTPUT_FILE = r"C:\TMU\postdoc-TMU\HammerDB\trace_dumps\lockname_query_results.txt"

# Define the output file for ROW locks
ROW_CSV_FILE = r"C:\TMU\postdoc-TMU\HammerDB\trace_dumps\row_locks.csv"
# Define the output file for TABLE locks
TABLE_CSV_FILE = r"C:\TMU\postdoc-TMU\HammerDB\trace_dumps\table_locks.csv"



# Regex patterns to extract data
#LOCKNAME_PATTERN_ENTRY = r"lockname\s+(\S+)"
LOCKNAME_PATTERN_ENTRY = r"lockname\s+(\S+)\s"
#LOCKNAME_PATTERN_EXIST = r"lockname\s+(\S+)"
LOCKNAME_PATTERN_EXIST = r"lockname\s+(\S+)\s"
#LOCKNAME_PATTERN_REM = r"Data3\s+\(PD_TYPE_SQLP_LOCKNAME,\d+\)\s+SQLP_LOCKNAME:\s+(\S+)"
#LOCKNAME_PATTERN_REM = r"lockname\s+(\S+)\s+SQLP_\S+"
LOCKNAME_PATTERN_REM = r"^(\S+)\s+SQLP_\w+\s+\(.*?\)$"
#LOCKNAME_PATTERN_REM_RECORD = r"(\S+)\s+SQLP_RECORD\s+\(obj=\{[^}]*\}\)"
TIMESTAMP_PATTERN = r"(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{9})"
DURATION_PATTERN = r"(\d+\.\d+)"  # Pattern to match duration value
CURINTENT_PATTERN = r"curIntent\s+(\S+)"  # Pattern to match curIntent value
INTENT_PATTERN = r"intent\s+(\S+)"  # Pattern to match intent value
MODE_PATTERN = r"mode\s+(\S+)"  # Pattern to match mode value

# Initialize DB2 connection
conn_str = f"DATABASE={db_name};HOSTNAME={db_host};PORT={db_port};PROTOCOL=TCPIP;UID={db_user};PWD={db_password};"

try:
    print("Trying to connect.")
    conn = ibm_db.connect(conn_str, "", "")
    print("Connected to the database successfully.")
except Exception as e:
    print(f"Failed to connect to the database: {str(e)}")
    #exit()

# Initialize dictionaries
lock_object_types = {}
lock_occurrences = {}
current_occurrence_id = {}

# Function to get the next occurrence ID for a lockname
def get_next_occurrence_id(lockname):
    if lockname not in lock_occurrences:
        lock_occurrences[lockname] = 1
    else:
        lock_occurrences[lockname] += 1
    return lock_occurrences[lockname]

# Function to get the current occurrence ID for a lockname
def get_current_occurrence_id(lockname):
    return lock_occurrences.get(lockname, 0)

# Function to execute DB2 query for lockname
def run_db2_query(lockname, original_lockname, start_timestamp, end_timestamp, start_duration,duration, curIntent, intent, mode):
    sql = f"SELECT SUBSTR(NAME, 1, 20) AS NAME, SUBSTR(VALUE, 1, 50) AS VALUE FROM TABLE(MON_FORMAT_LOCK_NAME('{original_lockname}')) AS LOCK"
    try:
        stmt = ibm_db.exec_immediate(conn, sql)
        results = []
        lock_type = None
        lock_details = {}
        while ibm_db.fetch_row(stmt):
            name = ibm_db.result(stmt, "NAME")
            value = ibm_db.result(stmt, "VALUE")
            if "LOCK_OBJECT_TYPE" in name:
                lock_type = value.strip()  # Remove any leading/trailing whitespace
            if lock_type in ["ROW", "TABLE"]:  # Check if lock_type is "ROW" or "TABLE"
                lock_details[name] = value
            results.append(f"Name: {name}, Value: {value}")
        
        # Write details to the text file
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"Lockname: {lockname}, Start Timestamp: {start_timestamp}, End Timestamp: {end_timestamp}, Duration: {duration}, curIntent: {curIntent}, intent: {intent}, mode: {mode}, Details: " + " | ".join(results) + "\n")

        # Update lock object types count
        if lock_type:
            if lock_type not in lock_object_types:
                lock_object_types[lock_type] = 0
            lock_object_types[lock_type] += 1

        # Write ROW lock details to CSV
        if lock_type == "ROW":
            write_row_lock_to_csv(lockname, original_lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode, lock_details)
                   # Write TABLE lock details to CSV
        elif lock_type == "TABLE":
            write_table_lock_to_csv(lockname, original_lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode, lock_details)


    except Exception as e:
        print(f"Error executing query for lockname {original_lockname}: {str(e)}")

def write_row_lock_to_csv(lockname, original_lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode, lock_details):
    file_exists = os.path.isfile(ROW_CSV_FILE)
    header = ['Original Lockname', 'Lockname', 'Start Timestamp', 'End Timestamp',  'Start_Duration', 'End_Duration', 'curIntent', 'intent', 'mode'] + list(lock_details.keys())
    with open(ROW_CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if the file does not exist
            writer.writerow(header)
        # Write row data
        row = [original_lockname, lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode] + [lock_details.get(key, '').strip() for key in header[9:]]
        writer.writerow(row)

def write_table_lock_to_csv(lockname, original_lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode, lock_details):
    file_exists = os.path.isfile(TABLE_CSV_FILE)
    header = ['Original Lockname', 'Lockname', 'Start Timestamp', 'End Timestamp',  'Start_Duration', 'End_Duration', 'curIntent', 'intent', 'mode'] + list(lock_details.keys())
    with open(TABLE_CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if the file does not exist
            writer.writerow(header)
        # Write row data
        row = [original_lockname, lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode] + [lock_details.get(key, '') for key in header[9:]]
        writer.writerow(row)

# Read input file line by line, extract locknames and timestamps, and execute DB2 queries
lock_data = {}  # Dictionary to store lockname details with start and end times
lock_type = None
current_lockname = None
current_lock = None
start_duration = None
duration = None
curIntent = None
intent = None
mode = None
timestamp_match = None

try:
    with open(INPUT_FILE, "r") as file:
        for line in file:
            # Set lock type based on line content
            if "sqlplrl entry" in line:
                lock_type = "sqlplrlEntry" # Ignore this data block

            if "sqlplrq exit" in line:
                lock_type = "sqlplrqExit"

            if lock_type == "sqlplrqExit":
                # Extract mode if present
                mode_match = re.search(MODE_PATTERN, line)
                if mode_match:
                    mode = mode_match.group(1)
                    print(f"Extracted mode: {mode}")  # Print the extracted mode

            if "sqlplrq entry" in line:
                lock_type = "entry"
                start_duration_match = re.search(DURATION_PATTERN, line)
                if start_duration_match:
                    start_duration = start_duration_match.group(1)
                    #print(f"Extracted start duration: {start_duration}")  # Print the extracted duration
            elif "sqlplrem entry" in line:
                lock_type = "rem"
                # Extract duration if present
                duration_match = re.search(DURATION_PATTERN, line)
                if duration_match:
                    duration = duration_match.group(1)
                    #print(f"Extracted duration: {duration}")  # Print the extracted duration

            # Time to filter out any not need blocks
            # Extract lockname based on lock type and manage occurrences
            if lock_type == "sqlplrlEntry" :
                continue
            
            #print(f"line: {line}")
   

            # Extract curIntent and intent if lock_type is "entry"
            if lock_type == "entry":
                curIntent_match = re.search(CURINTENT_PATTERN, line)
                if curIntent_match:
                    curIntent = curIntent_match.group(1)
                    #print(f"Extracted curIntent: {curIntent}")  # Print the extracted curIntent

                intent_match = re.search(INTENT_PATTERN, line)
                if intent_match:
                    intent = intent_match.group(1)
                    #print(f"Extracted intent: {intent}")  # Print the extracted intent

            #print(f"lock_type: {lock_type}")

            lockname_match = None
            if lock_type == "entry":
                lockname_match = re.search(LOCKNAME_PATTERN_ENTRY, line)
            elif lock_type == "rem":
                lockname_match = re.search(LOCKNAME_PATTERN_REM, line.strip())
               # if not lockname_match:
                #    lockname_match = re.search(LOCKNAME_PATTERN_REM_RECORD, line)
                #print(f"test match REM: {line}")
                #if lockname_match:
                    #print(f"correct REM: {line}")
                
            elif lock_type == "sqlplrqExit":
                lockname_match = re.search(LOCKNAME_PATTERN_EXIST, line)


            if lockname_match:
               
                lockname = lockname_match.group(1)
                print(f"Match lock: {lockname}")
                if lock_type == "entry":
                    occurrence_id = get_next_occurrence_id(lockname)
                else:
                    occurrence_id = get_current_occurrence_id(lockname)
                
                current_lockname = f"{lockname}_{occurrence_id}"
                if current_lockname not in lock_data and lock_type == "entry":
                    #print(f"Found lockname: {current_lockname}") 
                    lock_data[current_lockname] = {'start_timestamp': None, 'end_timestamp': None, 'curIntent': None, 'intent': None,  'start_duration': None,  'duration': None,'mode': None}
                try:
                    current_lock = lock_data[current_lockname]
                except KeyError:
                    print(f"KeyError: lockname '{current_lockname}' does not exist in lock_data, maybe and End REM Seen first.")
                    # Optionally, you can initialize current_lock with a default value if needed
                    #lock_data[current_lockname] = {'start_timestamp': None, 'end_timestamp': None, 'curIntent': None, 'intent': None, 'mode': None}
                    #current_lock = lock_data[current_lockname]
                except Exception as e:
                    print(f"Unexpected error occurred: {str(e)}")
                    # Handle other exceptions or re-raise if necessary

                        # Extract timestamp
            timestamp_match = re.search(TIMESTAMP_PATTERN, line)
            if timestamp_match and current_lock:
                timestamp = timestamp_match.group(1)
                if lock_type == "entry":
                    current_lock['start_timestamp'] = timestamp
                    current_lock['curIntent'] = curIntent
                    current_lock['start_duration'] = start_duration
                    current_lock['intent'] = intent
                elif lock_type == "rem":
                    current_lock['end_timestamp'] = timestamp
                    current_lock['duration'] = duration
                elif lock_type == "sqlplrqExit":    
                    current_lock['mode'] = mode


    # Process lock data
    for lockname_with_occurrence, times in lock_data.items():
        start_timestamp = times['start_timestamp']
        end_timestamp = times['end_timestamp']
        duration = times['duration']
        start_duration = times['start_duration']
        curIntent = times['curIntent']
        intent = times['intent']
        mode = times['mode']
        if start_timestamp and end_timestamp:
            original_lockname = lockname_with_occurrence.split('_')[0]
            print(f"Running query for lockname: {lockname_with_occurrence}, Start Time: {start_timestamp}, End Time: {end_timestamp}, start_duration: {start_duration},  Duration: {duration}, curIntent: {curIntent}, intent: {intent}, mode: {mode}")
            run_db2_query(lockname_with_occurrence, original_lockname, start_timestamp, end_timestamp, start_duration, duration, curIntent, intent, mode)

    print("Total number of unique locknames collected:", len(lock_data))
finally:
    # Close DB2 connection
    ibm_db.close(conn)
    print("Database connection closed.")

# Calculate and display statistics for LOCK_OBJECT_TYPE
total_locks = sum(lock_object_types.values())
print("\nStatistics for LOCK_OBJECT_TYPE:", total_locks)
for lock_type, count in lock_object_types.items():
    percentage = (count / total_locks) * 100
    print(f"Type: {lock_type}, Count: {count}, Percentage: {percentage:.2f}%")
