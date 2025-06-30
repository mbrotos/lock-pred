REM Step 5: Set the event monitors' state to 0 (inactive)
REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 SET EVENT MONITOR lock_event STATE = 0 && db2 SET EVENT MONITOR stmt_event STATE = 0 && db2 SET EVENT MONITOR uow_event STATE = 0  && db2 SET EVENT MONITOR deadlock_event STATE = 0  && db2 SET EVENT MONITOR connection_event STATE = 0 && db2 connect reset && db2 terminate"
REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 SET EVENT MONITOR lock_event STATE = 0  && db2 SET EVENT MONITOR uow_event STATE = 0  && db2 SET EVENT MONITOR deadlock_event STATE = 0  && db2 SET EVENT MONITOR connection_event STATE = 0 && db2 connect reset && db2 terminate"
if %errorlevel% neq 0 echo Step 5 failed

REM Step 6: Export the event monitor data to CSV files
REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO 'lock_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_LOCK_EVENT && db2 EXPORT TO 'stmt_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM STMT_STMT_EVENT && db2 EXPORT TO 'uow_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM UOW_UOW_EVENT && db2 connect reset && db2 terminate"
REM call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/lock_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_LOCK_EVENT && db2 EXPORT TO './data_csv_files/stmt_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM STMT_STMT_EVENT  && db2 connect reset && db2 terminate"
call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/lock_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_LOCK_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_ACTIVITY_VALUES_LOCK_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_ACTIVITY_VALUES_LOCK_EVENT  && db2 connect reset && db2 terminate"


call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANT_ACTIVITIES_LOCK_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANT_ACTIVITIES_LOCK_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANT_ACTIVITIES_LOCK_WAIT_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANT_ACTIVITIES_LOCK_WAIT_EVENT   && db2 connect reset && db2 terminate"



call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANTS_LOCK_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANTS_LOCK_EVENT  && db2 connect reset && db2 terminate"


call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANTS_LOCK_WAIT_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANTS_LOCK_WAIT_EVENT  && db2 connect reset && db2 terminate"




call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/CONTROL_LOCK_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONTROL_LOCK_EVENT  && db2 connect reset && db2 terminate"


call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/CONTROL_ACTIVITY_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONTROL_ACTIVITY_EVENT  && db2 connect reset && db2 terminate"


call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/CONTROL_LOCK_WAIT_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONTROL_LOCK_WAIT_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/CONTROL_UOW_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONTROL_UOW_EVENT  && db2 connect reset && db2 terminate"


 



call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/ACTIVITYMETRICS_ACTIVITY_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM ACTIVITYMETRICS_ACTIVITY_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/ACTIVITYSTMT_ACTIVITY_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM ACTIVITYSTMT_ACTIVITY_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/ACTIVITYVALS_ACTIVITY_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM ACTIVITYVALS_ACTIVITY_EVENT  && db2 connect reset && db2 terminate"



call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_LOCK_WAIT_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_LOCK_WAIT_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/DEADLOCK_DEADLOCK_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM DEADLOCK_DEADLOCK_EVENT  && db2 connect reset && db2 terminate"


 
call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCKEVMON_EVENT_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCKEVMON  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_LOCKEVMON_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_LOCKEVMON  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANTS_LOCKEVMON_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANTS_LOCKEVMON  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_PARTICIPANT_ACTIVITIES_LOCKEVMON_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_PARTICIPANT_ACTIVITIES_LOCKEVMON  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/LOCK_ACTIVITY_VALUES_LOCKEVMON_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM LOCK_ACTIVITY_VALUES_LOCKEVMON  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/CONTROL_LOCKEVMON_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONTROL_LOCKEVMON  && db2 connect reset && db2 terminate"


 db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/MON_GET_LOCKS_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM TABLE(MON_GET_LOCKS(NULL, -2)) AS T && db2 connect reset && db2 terminate"


call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root   && db2 EXPORT TO './data_csv_files/uow_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM UOW_UOW_EVENT  && db2 connect reset && db2 terminate"

call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/activity_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM ACTIVITY_ACTIVITY_EVENT   && db2 connect reset && db2 terminate"

db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root &&   db2 EXPORT TO './data_csv_files/connection_event_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM CONN_CONNECTION_EVENT    && db2 connect reset && db2 terminate"





db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/mon_get_table_orders_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM TABLE(MON_GET_TABLE(NULL, NULL, -2)) WHERE TABSCHEMA = 'DB2ADMIN' AND TABNAME = 'ORDERS' && db2 connect reset && db2 terminate"

db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO './data_csv_files/all_tables_mon_get_table_data.csv' OF DEL MODIFIED BY NOCHARDEL SELECT * FROM TABLE(MON_GET_TABLE(NULL, NULL, -2))  WHERE TABSCHEMA = 'DB2ADMIN'   && db2 connect reset && db2 terminate"

db2cmd /c /w /i "db2 -tvf ./export_header.sql"
 

REM SELECT * FROM TABLE(MON_GET_LOCKS(NULL, -2)) AS T; 
REM select * from SYSIBMADM.LOCKS_HELD
REM SELECT TABNAME FROM SYSCAT.TABLES
REM LIST TABLES FOR SCHEMA DB2ADMIN 
REM LOCK_ACTIVITY_VALUES_LOCK_EVENT 
REM LOCK_LOCK_EVENT                  
REM LOCK_PARTICIPANT_ACTIVITIES_LO
REM LOCK_PARTICIPANTS_LOCK_EVENT
REM SELECT ROUTINENAME, ROUTINESCHEMA, SPECIFICNAME, ROUTINETYPE, CREATE_TIME FROM SYSCAT.ROUTINES WHERE ROUTINETYPE = 'P' ORDER BY CREATE_TIME ASC;
REM SELECT count(*) from DB2ADMIN.LOCKS_HELD_HISTORY

REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO  'header.csv' of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', '),'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='MON_GET_TABLE'   && db2 connect reset && db2 terminate"

REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO header1.csv of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', \'),\'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='ORDERS'"

REM db2cmd /c /w /i "db2 EXPORT TO  'C:/TMU/postdoc-TMU/HammerDB/header.csv' of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', '),'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='ORDERS'"

REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO header1.csv of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', '),'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='ORDERS'"
REM db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 EXPORT TO 'c:/TMU/postdoc-TMU/HammerDB/header1.csv' of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', '),'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='ORDERS'"
REM EXPORT TO c:/TMU/postdoc-TMU/HammerDB/header1.csv of del SELECT SUBSTR(REPLACE(REPLACE(XMLSERIALIZE(CONTENT XMLAGG(XMLELEMENT(NAME c,colname) ORDER BY colno) AS VARCHAR(1500)),'<C>',', '),'</C>',''),3) FROM syscat.columns WHERE tabschema='DB2ADMIN' and tabname='ORDERS'




if %errorlevel% neq 0 echo Step 6 failed
 