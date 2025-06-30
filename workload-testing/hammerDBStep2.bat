cd "C:\Program Files\HammerDB-4.10"

echo "Step 4: Run the second HammerDB script"
hammerdbcli auto C:\TMU\postdoc-TMU\HammerDB\hammerDB_configs\testTPROCC.tcl > C:\TMU\postdoc-TMU\HammerDB\hammerdb.log
if %errorlevel% neq 0 echo Step 4 failed