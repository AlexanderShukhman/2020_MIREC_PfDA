#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import string
import re
import math
sys.path.append("../")

OK = 0
PE = 4
WA = 5
CF = 6

fin1 = float(open(sys.argv[2], "r").read())
#ans = fin.read().replace("\n", " ").replace("[", "").replace("]", "").split() # answer from user


#utf8stdout = open(1, "w", encoding="utf-8", closefd=False) # fd 1 is stdout
fin2 = float(open(sys.argv[3], "r").read())
#res = fin.read().replace("\n", " ").replace("[", "").replace("]", "").split() # correct answer

# sys.stderr.write(str(ans) + "\n", file=utf8stdout)
# print("contestant answer is " + res)

#res = []
#for given, correct in zip(ans, res):
#    x, y = float(given), float(correct)
#    res.append(math.fabs(x - y))

if abs(fin1 - fin2) < 1e-5:
    sys.exit(OK)
else:
    sys.exit(WA)

