# 对文本的预处理，去除空行，在每一句的结尾都加上句号。

f = open('wangfeng.txt', 'r')
lines = f.readlines()
f.close()
new_lines = []

for i in range(len(lines)):
    if (lines[i] == '\n') | (lines[i] == '演唱：汪峰\n'):
        continue
    else:
        new_lines.append(lines[i][:-1] + '。\n')
f = open('new_wangfeng.txt', 'w')
f.writelines(new_lines)
f.close()