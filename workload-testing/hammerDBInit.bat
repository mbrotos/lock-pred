cd "C:\Program Files\HammerDB-4.10"

echo "Step 1: Drop the database if it exists"
call db2cmd /c /w /i "db2 connect reset && db2 terminate" || echo Step 1 failed
call db2cmd /c /w /i "db2 connect to testdb3 user db2admin using root && db2 force applications all && db2 disconnect testdb3 && db2 drop database testdb3 &&  db2 connect reset && db2 terminate" || echo Step 1 failed
 
echo "Step 2: Run the first HammerDB script"
call hammerdbcli auto C:\TMU\postdoc-TMU\HammerDB\hammerDB_configs\InitTPROCC.tcl