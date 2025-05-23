@echo off
echo 正在启动 MAA 应用程序...

:: 检查Python是否正在运行
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo 检测到Python进程正在运行，正在关闭...
    taskkill /F /IM python.exe
    timeout /t 2 /nobreak >nul
)

:: 启动 Flask 应用
echo 正在启动后端服务...
:: 使用python -u 参数确保输出不被缓存
start /B python -u app.py

:: 等待服务器启动，最多等待10秒
set /a count=0
:WAIT_LOOP
timeout /t 1 /nobreak >nul
set /a count+=1
curl -s http://127.0.0.1:8000 >nul 2>&1
if errorlevel 1 (
    if %count% lss 10 (
        echo 等待服务器启动... %count%/10
        goto WAIT_LOOP
    ) else (
        echo 错误：服务器启动失败！
        echo 请检查：
        echo 1. Python是否正确安装
        echo 2. 所有依赖是否已安装
        echo 3. 端口8000是否被占用
        echo 4. app.py中的主程序代码是否正确执行
        pause
        exit /b 1
    )
)

:: 打开默认浏览器访问应用
echo 正在打开浏览器...
start http://127.0.0.1:8000

echo.
echo 应用程序已成功启动！
echo 如果浏览器没有自动打开，请手动访问: http://127.0.0.1:8000
echo.
echo 按任意键退出此窗口...
pause >nul 