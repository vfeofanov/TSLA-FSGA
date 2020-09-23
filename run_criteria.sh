for db_name; do
	python experiment.py "$db_name" 20 20 metric sup_tree stab rsla tsla
done    	
