import java.util.*;
import java.io.*;
import java.text.*;

/**
 *  
 *  Artificial Inteligence - Programming Assignment #4 - Naive Bayes Text Classification
 *  Professor Ernest Davis
 *  @author Anh Tran - 5/2/2023
 *  Due: 5/3/2023 - 11AM (EST)
 *  
 */
public class textClassifier {
    // MAIN
    public static void main(String args[]){
        String corpusFileName = "";
        // first numOfEntries(int) is used for training, rest is used for testing
        int numOfEntries = -1;
        // store stopwords
        HashSet<String> stopWords = new HashSet<String>();
        // store biographies split into name, category, and biography 
        // (trainingBiographies[0], trainingBiographies[1], trainingBiographies[2])
        ArrayList<String[]> trainingSet_T = new ArrayList<String[]>();
        ArrayList<String[]> predictionSet_T_P = new ArrayList<String[]>();
        // store unique categories
        HashSet<String> categories_C = new HashSet<String>();
        HashSet<String> allWordsInTrainingSet = new HashSet<String>();
        // store occurences
        HashMap<String, Integer> occurences = new HashMap<String, Integer>();
        // store frequencies
        HashMap<String, String> frequencies = new HashMap<String, String>();
        // store probabilities 
        HashMap<String, Double> probabilities = new HashMap<String, Double>();
        // store negative log probabilities
        HashMap<String, Double> negativeLogProbabilities = new HashMap<String, Double>();
        // store all prediction biographies
        ArrayList<String> predictionBiographyNames = new ArrayList<String>();
        // store all prediction biographies' categories
        HashMap<String, String> predictionBiographyCategories = new HashMap<String, String>();
        // store all prediction biographies' descriptions
        HashMap<String, HashSet<String>> predictionBiographyDescriptions = new HashMap<String, HashSet<String>>();
        // store classifying probabilities for each biography
        HashMap<String, Double> classifyingProbabilities = new HashMap<String, Double>();
        // store classifying results for each biography
        HashMap<String, String> chosenCategories = new HashMap<String, String>();
        // store actual probabilities for each biography
        HashMap<String, Double> actualProb = new HashMap<String, Double>();

        // READ INPUT
        try {
            String[] readInput = args;
            corpusFileName = readInput[0].trim();
            numOfEntries = Integer.parseInt(readInput[1]);
            // read stop words and extract them into a hashset
            readStopWords(stopWords);
            // read corpus and extract biographies into an arraylist
            readCorpusForTraining(corpusFileName, numOfEntries, trainingSet_T, stopWords);

            // get categories
            getCategories(categories_C, trainingSet_T);
            // get category occurences
            getCategoryOccurences(occurences, categories_C, trainingSet_T);

            // get unique words in biography of each category
            getUniqueWordsOfEachCategory(categories_C, trainingSet_T, allWordsInTrainingSet);
            getWordOccurences(occurences, allWordsInTrainingSet, categories_C, trainingSet_T);

            // get category frequencies
            getCategoryFrequencies(frequencies, occurences, categories_C, trainingSet_T);
            // get word frequencies
            getWordFrequencies(frequencies, occurences, allWordsInTrainingSet, categories_C, trainingSet_T);

            // get category probability
            getCategoryProbabilities(probabilities, frequencies, categories_C, occurences);
            // get word probability
            getWordProbabilities(probabilities, frequencies, allWordsInTrainingSet, categories_C);

            // get category negative log probability
            getCategoryLogProbabilities(negativeLogProbabilities, probabilities, categories_C);
            // get word negative log probability
            getWordLogProbabilities(negativeLogProbabilities, probabilities, allWordsInTrainingSet, categories_C);
            // read corpus for prediction
            readCorpusForPrediction(corpusFileName, numOfEntries, predictionSet_T_P, stopWords, allWordsInTrainingSet);

            // get all prediction biographies
            getAllPredictionBiographyNames(predictionBiographyNames, predictionBiographyCategories, 
            predictionBiographyDescriptions, predictionSet_T_P);

            // get classifying probabilities
            getCategoryProbabilitiesForPrediction(predictionBiographyNames, predictionBiographyDescriptions, 
            negativeLogProbabilities, categories_C, classifyingProbabilities, chosenCategories);
            
            // recover actual probabilities
            recoverActualProbabilities(actualProb, classifyingProbabilities, predictionBiographyNames,
            categories_C, chosenCategories);

            // export results
            exportResults("Output_" + corpusFileName, predictionBiographyNames, predictionBiographyCategories, 
            chosenCategories, actualProb, categories_C);
            
        } catch (Exception e) {
            System.err.println("Error: Please follow the command line format: `java textClassifier.java <corpusFileName> <numOfEntries>`");
            System.exit(0);
        }
        return;
    }

    // READ AND EXTRACT STOP WORDS
    public static void readStopWords(HashSet<String> stopWords) throws FileNotFoundException{
        try {
            File stopWordsInputFile = new File("stopwords.txt");
            Scanner scanner = new Scanner(stopWordsInputFile);
            while(scanner.hasNextLine()){
                String line = scanner.nextLine().trim();
                if(line.equals("")){
                    continue;
                }
                else{
                    String[] wordsInLine = line.split("\\s+");
                    for(String word : wordsInLine){
                        stopWords.add(word.toLowerCase());
                    }
                }
            }
            scanner.close();
        } catch (Exception e) {
            System.err.println("Error: Error reading 'stopwords.txt' file.");
            System.exit(0);
        }
        return;
    }

    // READ AND EXTRACT BIOGRAPHIES
    public static void readCorpusForTraining(String corpusFileName, int numOfTrainingEntries, 
    ArrayList<String[]> trainingBiographies, HashSet<String> stopWords) throws FileNotFoundException{
        try {
            File corpusInputFile = new File(corpusFileName);
            Scanner scanner = new Scanner(corpusInputFile);
            // for each biography to read in corpus
            for (int i = 0 ; i < numOfTrainingEntries; i++){
                String[] biography = new String[3];
                // read name
                String nameLine = scanner.nextLine().trim();
                while(nameLine.equals("")){
                    nameLine = scanner.nextLine().trim();
                }
                biography[0] = nameLine;
                // read category
                String categoryLine = scanner.nextLine().trim().toLowerCase();
                biography[1] = categoryLine;
                // read biography
                String biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                String biographyDescription = "";
                while(!biographyDescriptionLine.equals("")){
                    String[] wordsInLine = biographyDescriptionLine.split("\\s+");
                    // eliminate stop words and words with length 1 or 2
                    for (int j = 0; j < wordsInLine.length; j++){
                        if(stopWords.contains(wordsInLine[j]) || wordsInLine[j].length() <= 2){
                            wordsInLine[j] = "";
                        }
                        else{
                            wordsInLine[j] = wordsInLine[j].replaceAll("[^a-zA-Z ]", "");
                        }
                    }
                    // eliminate empty strings
                    for (int j = 0; j < wordsInLine.length; j++){
                        if(wordsInLine[j].equals("")){
                            continue;
                        }
                        else{
                            biographyDescription += wordsInLine[j] + " ";
                        }
                    }
                    biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                }
                biography[2] = biographyDescription.trim().toLowerCase();
                // add biography to arraylist
                trainingBiographies.add(biography);
            }
            scanner.close();
        } catch (Exception e) {
            System.err.println("Error: Error reading " + corpusFileName + " file.");
            System.exit(0);
        }
        return;
    }

    // GET CATEGORIES
    public static void getCategories(HashSet<String> categories, ArrayList<String[]> trainingSet_T){
        for (String[] biography : trainingSet_T){
            categories.add(biography[1]);
        }
        return;
    }

    // GET CATEGORY OCCURENCES
    public static void getCategoryOccurences(HashMap<String, Integer> categoryOccurences, HashSet<String> categories, 
    ArrayList<String[]> trainingSet_T){
        for (String category : categories){
            int occurences = 0;
            for (String[] biography : trainingSet_T){
                if (biography[1].equals(category)){
                    occurences++;
                }
            }
            categoryOccurences.put(category, occurences);
        }
        return;
    }

    // GET UNIQUE WORDS IN BIOGRAPHY OF EACH CATEGORY
    public static void getUniqueWordsOfEachCategory(HashSet<String> categories, 
    ArrayList<String[]> trainingSet_T, HashSet<String> allWordsInTrainingSet){
        for (String category : categories){
            for (String[] biography : trainingSet_T){
                if (biography[1].equals(category)){
                    String[] wordsInBiography = biography[2].split("\\s+");
                    for (String word : wordsInBiography){
                        allWordsInTrainingSet.add(word);
                    }
                }
            }
        }
        return;
    }

    // GET WORD OCCURENCES IN BIOGRAPHY OF EACH CATEGORY
    public static void getWordOccurences(HashMap<String, Integer> wordOccurences, HashSet<String> uniqueWords, 
    HashSet<String> categories_C, ArrayList<String[]> trainingSet_T){
        // for each category
        for (String category : categories_C){
            // for each word in training set
            for (String word : uniqueWords){
                String wordCategory = word + " | " + category;
                int occurences = 0;
                // for each biography
                for (String[] biography : trainingSet_T){
                    String biographyCategory = biography[1];
                    if (!biographyCategory.equals(category)){
                        continue;
                    }
                    else{
                        String[] biographyWords = biography[2].trim().split("\\s+");
                        for (String biographyWord : biographyWords){
                            if (biographyWord.equals(word)){
                                occurences++;
                                break;
                            }
                        }
                    }
                }
                wordOccurences.put(wordCategory, occurences);
            }
        }
        return;
    }

    // GET CATEGORY FREQUENCIES
    public static void getCategoryFrequencies( HashMap<String, String> frequencies, HashMap<String, Integer> occurences,
    HashSet<String> categories_C, ArrayList<String[]> trainingSet_T){
        // for each category
        for (String category : categories_C){
            int numerator = occurences.get(category);
            int denominator = trainingSet_T.size();
            String frequency = Integer.toString(numerator) + "/" + Integer.toString(denominator);
            frequencies.put(category, frequency);
        }
        return;
    }

    // GET WORD FREQUENCIES
    public static void getWordFrequencies(HashMap<String, String> frequencies, HashMap<String, Integer> occurences,
    HashSet<String> uniqueWords, HashSet<String> categories_C, ArrayList<String[]> trainingSet_T){
        // for each category
        for (String category : categories_C){
            // for each word in training set
            for (String word : uniqueWords){
                String wordCategory = word + " | " + category;
                int numerator = occurences.get(wordCategory);
                int denominator = occurences.get(category);
                String frequency = Integer.toString(numerator) + "/" + Integer.toString(denominator);
                frequencies.put(wordCategory, frequency);
            }
        }
        return;
    }

    // GET CATEGORY PROBABILITIES
    public static void getCategoryProbabilities(HashMap<String, Double> probabilities, HashMap<String, String> frequencies,
    HashSet<String> categories_C, HashMap<String, Integer> occurences){
        // for each category
        for (String category : categories_C){
            String frequency = frequencies.get(category);
            String[] fraction = frequency.split("/");
            double freqNumerator = Double.parseDouble(fraction[0]);
            double freqDenominator = Double.parseDouble(fraction[1]);
            double frequencyInDecimal = freqNumerator / freqDenominator;
            double probabilityNumerator = frequencyInDecimal + 0.1;
            double noOfCategories = occurences.get(category);
            double probabilityDenominator = 1 + noOfCategories * 0.1;
            double probability = probabilityNumerator / probabilityDenominator;
            probabilities.put(category, probability);
        }
        return;
    }

    // GET WORD PROBABILITIES
    public static void getWordProbabilities(HashMap<String, Double> probabilities, HashMap<String, String> frequencies,
    HashSet<String> uniqueWords, HashSet<String> categories_C){
        // for each category
        for (String category : categories_C){
            // for each word in training set
            for (String word : uniqueWords){
                String wordCategory = word + " | " + category;
                String frequency = frequencies.get(wordCategory);
                String[] fraction = frequency.split("/");
                double freqNumerator = Double.parseDouble(fraction[0]);
                double freqDenominator = Double.parseDouble(fraction[1]);
                double frequencyInDecimal = freqNumerator / freqDenominator;
                double probabilityNumerator = frequencyInDecimal + 0.1;
                double probabilityDenominator = 1 + 2 * 0.1;
                double probability = probabilityNumerator / probabilityDenominator;
                probabilities.put(wordCategory, probability);
            }
        }
        return;
    }
    
    // GET CATEGORY LOG PROBABILITIES
    public static void getCategoryLogProbabilities(HashMap<String, Double> logProbabilities, HashMap<String, Double> probabilities,
    HashSet<String> categories_C){
        // for each category
        for (String category : categories_C){
            double probability = probabilities.get(category);
            double logProbability = -(Math.log(probability) / Math.log(2));
            logProbabilities.put(category, logProbability);
        }
        return;
    }

    // GET WORD LOG PROBABILITIES
    public static void getWordLogProbabilities(HashMap<String, Double> logProbabilities, HashMap<String, Double> probabilities,
    HashSet<String> uniqueWords, HashSet<String> categories_C){
        // for each category
        for (String category : categories_C){
            // for each word in training set
            for (String word : uniqueWords){
                String wordCategory = word + " | " + category;
                double probability = probabilities.get(wordCategory);
                double logProbability = -(Math.log(probability) / Math.log(2));
                logProbabilities.put(wordCategory, logProbability);
            }
        }
        return;
    }

    // READ CORPUS FOR PREDICTION
    public static void readCorpusForPrediction(String corpusFileName, int numOfTrainingEntries, 
    ArrayList<String[]> predictionBiographies, HashSet<String> stopWords, 
    HashSet<String> allWordsInTrainingSet) throws FileNotFoundException{
        try {
            File corpusInputFile = new File(corpusFileName);
            Scanner scanner = new Scanner(corpusInputFile);
            // skip training entries
            for (int i = 0 ; i < numOfTrainingEntries; i++){
                // read name
                String nameLine = scanner.nextLine().trim();
                while(nameLine.equals("")){
                    nameLine = scanner.nextLine().trim();
                }
                // read category
                scanner.nextLine().trim().toLowerCase();
                // read biography
                String biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                while(!biographyDescriptionLine.equals("")){
                    biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                    continue;
                }
            }
            // prediction biography entries
            while(scanner.hasNextLine()){
                
                String[] biography = new String[3];
                // read name
                String nameLine = scanner.nextLine().trim();
                
                while(nameLine.equals("")){
                    nameLine = scanner.nextLine().trim();
                }
                biography[0] = nameLine;
                // read category
                String categoryLine = scanner.nextLine().trim().toLowerCase();
                biography[1] = categoryLine;
                
                // read biography
                String biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                String biographyDescription = "";
                while(!biographyDescriptionLine.equals("")){
                    String[] wordsInLine = biographyDescriptionLine.split("\\s+");
                    // eliminate stop words and words with length 1 or 2 
                    // and any word that did not appear in training set
                    for (int j = 0; j < wordsInLine.length; j++){
                        if(stopWords.contains(wordsInLine[j]) || wordsInLine[j].length() <= 2 ){
                            wordsInLine[j] = "";
                        }
                        else{
                            wordsInLine[j] = wordsInLine[j].replaceAll("[^a-zA-Z ]", "");
                            if(!(allWordsInTrainingSet.contains(wordsInLine[j]))){
                                wordsInLine[j] = "";   
                            }
                        }
                    }
                    // eliminate empty strings
                    for (int j = 0; j < wordsInLine.length; j++){
                        if(wordsInLine[j].equals("")){
                            continue;
                        }
                        else{
                            biographyDescription += wordsInLine[j] + " ";
                        }
                    }
                    if(scanner.hasNextLine()){
                        biographyDescriptionLine = scanner.nextLine().trim().toLowerCase();
                    }
                    else{
                        break;
                    }
                }
                biography[2] = biographyDescription.trim().toLowerCase();
                // add biography to arraylist
                predictionBiographies.add(biography);
            }
            scanner.close();
        } catch (Exception e) {
            System.err.println("Error: Error reading " + corpusFileName + " file.");
            System.exit(0);
        }
        return;
    }

    // MAP PREDICTION BIOGRAPHY NAMES TO CATEGORIES AND DESCRIPTIONS
    public static void getAllPredictionBiographyNames(ArrayList<String> biographyNames, HashMap<String, String> predictionBiographyCategories, 
    HashMap<String, HashSet<String>> predictionBiographyDescriptions, ArrayList<String[]> predictionBiographies){
        for (String[] biography : predictionBiographies){
            String name = biography[0];
            String category = biography[1];
            String[] description = biography[2].trim().split("\\s+");
            HashSet<String> descriptionWordSet = new HashSet<String>();
            for(String word : description){
                descriptionWordSet.add(word);
            }
            biographyNames.add(name);
            predictionBiographyCategories.put(name, category);
            predictionBiographyDescriptions.put(name, descriptionWordSet);
        }
        return;
    }

    // GET CATEGORY PROBABILITIES FOR PREDICTION
    public static void getCategoryProbabilitiesForPrediction(ArrayList<String> predictionBiographyNames, 
    HashMap<String, HashSet<String>> predictionBiographyDescriptions, HashMap<String, Double> logProbabilities, 
    HashSet<String> categories_C, HashMap<String, Double> classifyingProbabilities, 
    HashMap<String, String> chosenCategories){
        // for each biography
        for (String name : predictionBiographyNames){
            double chosenProb = Double.POSITIVE_INFINITY;
            String chosenCategory = "";
            // for each category
            for (String category : categories_C){
                String categoryBiography = category + " | " + name;
                double lc = logProbabilities.get(category);
                HashSet<String> uniqueWordsInDescription = predictionBiographyDescriptions.get(name);
                Double sum = 0.0;
                for (String word : uniqueWordsInDescription){
                    String wordCategory = word + " | " + category;
                    if(!logProbabilities.containsKey(wordCategory)){
                        continue;
                    }
                    Double lw = logProbabilities.get(wordCategory);
                    sum += lw;
                }
                
                double probability = lc + sum;
                classifyingProbabilities.put(categoryBiography, probability);
                if (probability < chosenProb){
                    chosenProb = probability;
                    chosenCategory = category;
                }
            }
            chosenCategories.put(name, chosenCategory);
        }
        return;
    }

    // RECOVER ACTUAL PROBABILITIES
    public static void recoverActualProbabilities(HashMap<String, Double> actualProb,
    HashMap<String, Double> classifyingProbabilities, ArrayList<String> biographyNames,
    HashSet<String> categories_C, HashMap<String, String> chosenCategories){
        HashMap<String, Double> XI = new HashMap<String, Double>();
        // for each biography
        for (String name : biographyNames){
            String chosenCategory = chosenCategories.get(name);
            String chosenCategoryBiography = chosenCategory + " | " + name;
            double m = classifyingProbabilities.get(chosenCategoryBiography);
            double s = 0.0;
            for (String category : categories_C){
                String categoryBiography = category + " | " + name;
                
                double ci = classifyingProbabilities.get(categoryBiography);
                
                if (ci - m < 7){
                    double probability = Math.pow(2, m - ci);
                    s += probability;
                    XI.put(categoryBiography, probability);
                }
                else{
                    XI.put(categoryBiography, 0.0);
                }
            }
            for (String category : categories_C){
                String categoryBiography = category + " | " + name;
                double probability = XI.get(categoryBiography);
                double actualProbability = probability / s;
                actualProb.put(categoryBiography, actualProbability);
            }
        }
        return;
    }

    // EXPORT RESULTS
    public static void exportResults(String outputFileName, ArrayList<String> biographyNames,
    HashMap<String, String> predictionBiographyCategories, HashMap<String, String> chosenCategories,
    HashMap<String, Double> actualProb, HashSet<String> categories_C){
        try {
            DecimalFormat df = new DecimalFormat("#.##");
            File outputFile = new File(outputFileName);
            FileWriter fileWriter = new FileWriter(outputFile);
            PrintWriter printWriter = new PrintWriter(fileWriter);
            int totalPredictions = 0;
            int correctPredictions = 0;
            for (String name : biographyNames){
                String category = predictionBiographyCategories.get(name);
                String chosenCategory = chosenCategories.get(name);
                String result = "Wrong.\n";
                totalPredictions+=1;
                if (category.equals(chosenCategory)){
                    result = "Correct.\n";
                    correctPredictions+=1;
                }
                String firstLine = name + ".    Prediction: " + chosenCategory.substring(0,1).toUpperCase() + 
                chosenCategory.substring(1).toLowerCase() + ".    " + result;
                String secondLine = "";
                for(String category_C : categories_C){
                    String categoryBiography = category_C + " | " + name;
                    double actualProbability = actualProb.get(categoryBiography);
                    secondLine += category_C.substring(0,1).toUpperCase() + 
                    category_C.substring(1).toLowerCase() + ": " + df.format(actualProbability) + ".    ";
                }
                secondLine = secondLine.trim() + "\n";
                printWriter.println(firstLine + secondLine);
            }
            double correctPercentage = (double) correctPredictions / (double) totalPredictions;
            printWriter.println("Overall accuracy: " + correctPredictions + " out of " 
            + totalPredictions + " = " + df.format(correctPercentage) + ".");
            printWriter.close();
        } catch (Exception e) {
            System.err.println("Error: Error writing to " + outputFileName + " file.");
            System.exit(0);
        }
        return;
    }
}