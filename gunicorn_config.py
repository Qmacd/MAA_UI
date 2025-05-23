import multiprocessing

# 工作进程数
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式
worker_class = 'gevent'

# 绑定地址
bind = '0.0.0.0:7000'

# 超时时间
timeout = 300

# 最大请求数
max_requests = 1000

# 最大请求抖动
max_requests_jitter = 50

# 工作进程名称
proc_name = 'maa_ui'

# 访问日志
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'

# 守护进程模式
daemon = True

# 进程ID文件
pidfile = 'logs/gunicorn.pid' 