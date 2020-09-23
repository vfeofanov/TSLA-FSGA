for db_name; do
	python experiment.py "$db_name" 20 20 mate_type standard weighted
	python experiment.py "$db_name" 20 20 max_num_mut "1-2"
	python experiment.py "$db_name" 20 20 relevance_test 1
done    	
