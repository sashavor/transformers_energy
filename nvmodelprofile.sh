#/bin/bash
#
#
# Log for 60 seconds into output.log
#
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1 -f "output.log"  
