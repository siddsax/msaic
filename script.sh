cd data/TrainData
tp="temp"
for file in *.txt
do
  echo $file
  shuf $file > $tp
  cat $tp > $file
done





