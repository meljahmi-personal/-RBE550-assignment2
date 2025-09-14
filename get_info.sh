echo "Operating System:" > environment_info.txt
uname -a >> environment_info.txt
echo "" >> environment_info.txt
echo "Python Version:" >> environment_info.txt
python3 --version >> environment_info.txt
echo "" >> environment_info.txt
echo "Installed Packages:" >> environment_info.txt
pip freeze > requirements.txt

