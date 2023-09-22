
for id in $(seq 1 359)
do  
    echo $id
    python facescape2renderme.py ${id}
done