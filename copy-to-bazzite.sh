#!/bin/bash
# rsync -arzhv0 ./datasets/pursuit/barq/ user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/datasets/pursuit/barq/
rsync -arzhv0 ./barq_cache.db* user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/
rsync --ignore-existing -arzhv0 ./datasets/ user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/datasets

