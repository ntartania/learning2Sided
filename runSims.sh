SECONDS=0
source ~/.bashrc
kinit && aklog

echo "######### Starting ######### "
echo "1/10"
python Sim.py -i parameters/parameters.config

echo "2/10"
python Sim.py -i parameters/parameters2.config

echo "3/10"
python Sim.py -i parameters/parameters3.config

echo "4/10"
python Sim.py -i parameters/parameters4.config

echo "5/10"
python Sim.py -i parameters/parameters5.config

echo "6/10"
python Sim.py -i parameters/parameters6.config

echo "7/10"
python Sim.py -i parameters/parameters7.config

echo "8/10"
python Sim.py -i parameters/parameters8.config

echo "9/10"
python Sim.py -i parameters/parameters9.config

echo "10/10"
python Sim.py -i parameters/parameters10.config

echo "######### Finished #########"

if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
