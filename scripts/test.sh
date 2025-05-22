vessel="LV"
test_break="file1900"
#test_path = path/to/your/data
test_path="test_example/$vessel/$test_break"
rotate_phi=1
separate_files=0

python3 test.py "$vessel" "$test_break" "$test_path" "$rotate_phi" "$separate_files"
