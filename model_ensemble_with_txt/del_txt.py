path = 'final_file'
name = 'adapt_pred'#txt file after model ensemble
with open("./%s/txt/%s.txt" % (path,name),"r",encoding="utf-8") as f:
    lines = f.readlines()
with open("./%s/txt/%s.txt" % (path,name),"w",encoding="utf-8") as f_w:
    for line in lines:
        line_list = line.split("/")
        line_list = line_list[6:]
        line = '/'.join(line_list)
        f_w.write(line)