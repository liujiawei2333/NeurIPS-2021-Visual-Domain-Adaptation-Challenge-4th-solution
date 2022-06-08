import csv
import openpyxl
import os

path = 'final_file'
name = 'adapt_pred_'
num = 'B6'

if not os.path.exists('%s/xlsx' % path):
    os.makedirs('%s/xlsx' % path)

input_file = './%s/txt/%s%s.txt' % (path,name,num)
output_file = './%s/xlsx/%s%s.xlsx' % (path,name,num)
wb = openpyxl.Workbook()
ws = wb.worksheets[0]

with open(input_file,'rt',encoding="utf-8") as data:
    reader = csv.reader(data,delimiter=' ')
    for row in reader:
        ws.append(row)

wb.save(output_file)
