@echo off
echo 正在启动 MAA 应用程序...

:: 启动 Flask 应用
start /B python app.py

:: 等待服务器启动
timeout /t 2 /nobreak >nul

:: 打开默认浏览器访问应用
start http://127.0.0.1:8000

echo 应用程序已启动！
echo 如果浏览器没有自动打开，请手动访问: http://127.0.0.1:8000
echo 按任意键退出此窗口...
pause >nul 