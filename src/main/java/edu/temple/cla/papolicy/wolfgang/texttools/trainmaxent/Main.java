/* 
 * Copyright (c) 2018, Temple University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * All advertising materials features or use of this software must display 
 *   the following  acknowledgement
 *   This product includes software developed by Temple University
 * * Neither the name of the copyright holder nor the names of its 
 *   contributors may be used to endorse or promote products derived 
 *   from this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
package edu.temple.cla.papolicy.wolfgang.texttools.trainmaxent;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.temple.cla.papolicy.wolfgang.texttools.util.CommonFrontEnd;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Util;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Vocabulary;
import edu.temple.cla.papolicy.wolfgang.texttools.util.WordCounter;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import picocli.CommandLine;

/**
 * This is a front end that is compatible with the other TextTools for a
 * trainer of a Maximum Entropy classifier. The actual training is performed
 * by the <a href = "https://nlp.stanford.edu/software/classifier.shtml">
 * Stanford Classifier</a>.
 * @author Paul Wolfgang
 */
public class Main implements Callable<Void> {

    @CommandLine.Option(names = "--model", description = "Directory where model files are written")
    private String modelOutput = "Model_Dir";

    private final String[] args;

    
    public Main(String[] args) {
        this.args = args;
    }

    /**
     * Main entry point.
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Main main = new Main(args);
        CommandLine commandLine = new CommandLine(main);
        commandLine.setUnmatchedArgumentsAllowed(true).parse(args);
        try {
            main.call();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Actual code of the main method. This code is called after the command-
     * line parameters have been populated into the fields.
     * @return null.
     * @throws Exception as specified in the Callable interface.
     */
    @Override
    public Void call() throws Exception {
        try {
            List<Map<String, Object>> cases = new ArrayList<>();
            CommonFrontEnd commonFrontEnd = new CommonFrontEnd();
            CommandLine commandLine = new CommandLine(commonFrontEnd);
            commandLine.setUnmatchedArgumentsAllowed(true);
            commandLine.parse(args);
            Vocabulary vocabulary = commonFrontEnd.loadData(cases);
            File modelParent = new File(modelOutput);
            Util.delDir(modelParent);
            modelParent.mkdirs();
            Util.outputFile(modelParent, "vocab.bin", vocabulary);
            GeneralDataset<String, String> trainingData = createDataset(cases);
            Classifier<String, String> classifier = makeClassifier(trainingData);
            Util.outputFile(modelParent, "classifier.bin", classifier);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }
       
    /**
     * Method to create the Dataset object to be used by the trainer.
     * @param cases The training cases
     * @return A Dataset object containing the labels and features.
     */
    public Dataset<String, String> createDataset(List<Map<String, Object>> cases) {
        Dataset<String, String> dataSet = new Dataset<>(cases.size());
        cases.forEach(trainingCase -> {
            WordCounter count = (WordCounter)trainingCase.get("counts");
            dataSet.add(count.getWords(), trainingCase.get("theCode").toString());
        });
        return dataSet;
    }
    
    /**
     * Train the classifier.
     * @param trainingData The training data.
     * @return A trained classifier.
     */
    public Classifier<String, String> makeClassifier(GeneralDataset<String, String> trainingData) {
        Classifier<String, String> lc;
        LinearClassifierFactory<String, String> lcf = new LinearClassifierFactory<>();
        lc = lcf.trainClassifier(trainingData);
        return lc;
    }

}
