import numpy as np
cat2real=[-1,121,134,153,-1,-1,17,-1,111,56,-1,119,84,242,187,135,130,87,90,182,243,55,313,-1,106,-1,-1,-1,234,-1,73,125,80,51,66,154,-1,2,131,67,22,117,30,128,68,28,136,66,-1,135,89,276,33,-1,81,36,150,113,-1,11,-1,72,151,-1,52,74,52,-1,46,-1,205,19,2,227,161,3,-1,110,53,74,-1,213,72,61,-1,39,81,2,-1,52,170,69,153,25,-1,81,74,74,-1,-1,-1,32,126,88,-1,33,130,41,-1,181,99,14,175,-1,-1,74,236,100,183,134,9,6,224,297,-1,-1,14,-1,48,93,68,153,-1,6,-1,13,105,-1,16,112,-1,124,57,153,118,-1,188,168,-1,-1,-1,172,67,257,-1,122,-1,-1,22,85,-1,78,130,92,-1,200,-1,-1,49,167,-1,67,-1,-1,195,-1,158,-1,68,228,-1,-1,7,-1,296,37,141,147,148,86,70,90,230,78,66,12,260,318,199,1,144,83,15,152,176,231,159,139,26,268,43,-1,201,5,240,68,214,188,95,42,4,-1,9,155,97,27,145,112,101,-1,91,130,9,-1,-1,52,251,132,138,60,74,74,217,68,-1,-1,20,192,35,16,-1,-1,74,-1,76,40,8,79,-1,38,50,207,-1,156,29,209,-1,-1,98,246,91,109,157,-1,96,90,156,142,114,115,206,66,58,58,-1,44,104,45,94,-1,123,82,44,-1,9,104,31,68,-1,-1,24,51,305,108,34,-1,-1,109,160,143,98,-1,21,77,158,-1,202,137,8,-1,-1,-1,62,133,-1,186,-1,140,-1,-1,65,-1,78,223,25,120,59,102,146,103,194,-1,116,-1,-1,-1,10,165,75,-1,71,136,109,-1,23,149,162,177,-1,59,190,-1,-1,-1,191,180]
#cat2real=[-1,121,210,153,-1,-1,17,-1,111,-1,-1,119,84,52,187,135,16,87,259,-1,-1,-1,137,-1,9,-1,-1,-1,234,-1,-1,-1,-1,51,66,-1,-1,54,-1,67,64,-1,30,128,68,28,285,-1,-1,65,-1,276,33,-1,-1,-1,153,-1,-1,11,-1,72,151,-1,52,74,-1,-1,46,-1,-1,19,-1,-1,161,3,-1,110,53,74,-1,-1,72,61,-1,-1,-1,2,-1,168,170,-1,-1,-1,-1,81,-1,74,-1,-1,-1,32,60,88,-1,33,130,41,-1,-1,139,14,-1,-1,-1,74,236,-1,183,-1,9,123,-1,121,-1,-1,-1,-1,48,93,-1,153,-1,6,-1,13,105,-1,16,271,-1,124,-1,153,118,-1,-1,341,-1,-1,-1,172,67,257,-1,122,-1,-1,22,-1,-1,24,130,92,-1,200,-1,-1,49,-1,-1,67,-1,-1,195,-1,-1,-1,108,228,-1,-1,-1,-1,-1,-1,141,-1,-1,-1,-1,90,-1,78,66,12,-1,74,199,1,5,83,7,152,176,-1,159,140,42,268,-1,-1,-1,5,-1,68,214,188,-1,42,4,-1,-1,155,97,-1,145,112,101,-1,-1,130,9,-1,-1,52,251,132,138,-1,249,74,-1,68,-1,-1,8,192,35,16,-1,-1,74,-1,-1,-1,-1,-1,-1,38,50,207,-1,-1,29,-1,-1,-1,98,-1,91,109,-1,-1,96,-1,156,-1,-1,-1,-1,66,58,-1,-1,44,-1,45,94,-1,123,82,-1,-1,9,16,31,68,-1,-1,24,-1,-1,108,-1,-1,-1,109,160,-1,350,-1,21,77,158,-1,-1,-1,8,-1,-1,-1,62,133,-1,-1,-1,140,-1,-1,65,-1,78,223,110,120,59,102,146,103,194,-1,116,-1,-1,-1,10,-1,75,-1,71,136,109,-1,23,-1,-1,1,-1,9,-1,-1,-1,-1,160,180]
