@echo off
echo 正在设置start_app.bat的图标...

:: 创建VBS脚本来设置图标
echo Set oWS = WScript.CreateObject("WScript.Shell") > SetIcon.vbs
echo sLinkFile = "start_app.bat.lnk" >> SetIcon.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> SetIcon.vbs
echo oLink.TargetPath = "%~dp0start_app.bat" >> SetIcon.vbs
echo oLink.IconLocation = "%~dp0maa_icon.ico" >> SetIcon.vbs
echo oLink.Save >> SetIcon.vbs

:: 运行VBS脚本
cscript //nologo SetIcon.vbs

:: 清理临时文件
del SetIcon.vbs

echo 图标设置完成！
echo 现在start_app.bat的快捷方式已经使用了新的图标。
pause