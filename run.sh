K="4 8 16 32 64"
for k in $K:
do
	python getData.py --k $k
done
