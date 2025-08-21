import psutil
a = 0
while a < 10:
    a += 1
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f'CPU usage: {cpu_percent}%')
