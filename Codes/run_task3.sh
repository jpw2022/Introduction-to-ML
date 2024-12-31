for file in ./config_files/task3/*.config; do
	file_name="${file#./config_files/}"
	echo "running $file_name"
	python3 main.py -f "$file_name"
done
