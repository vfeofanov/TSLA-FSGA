for db_name; do
	python experiment.py "$db_name" 20 20 sota rlsr sfs ssls co_train_fss tsla_fss
done    	
