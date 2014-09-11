package engine;

import mulan.data.MultiLabelInstances;

public class Engine {
	
	private static int CVfolds = 2;
	private static String evaluationMeasure = "Hamming Loss";

    public static void main(String[] args) throws Exception {
    	
    	// check for data-set argument
    	if (args.length != 2) {
	        System.out.println("Lacking dataset and header as attributes.\n");
	        System.exit(1);
	     }

		// load data-set
    	System.out.println("Reading dataset...");
    	MultiLabelInstances dataset = new MultiLabelInstances(args[0], args[1]);
    	
    	// use data-set with *CVfolds*-folds cross-validation
    	Learners learn = new Learners(dataset, CVfolds);
    	learn.evaluate(evaluationMeasure);
    	
    	// Gather HammingLoss results and print them
    	System.out.println(learn.toString());
    	
    	// Exit message
	    System.out.println("Done!");
	}
	
}
