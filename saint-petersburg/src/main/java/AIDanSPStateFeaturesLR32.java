import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import smile.classification.LogisticRegression;

public class AIDanSPStateFeaturesLR32 {
    String modelFilename = "AIDanSPLogisticRegression32.model";
    LogisticRegression.Binomial model;
    ArrayList<SPFeature> features;

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public AIDanSPStateFeaturesLR32() {
    	features = new ArrayList<>();
        features.add(new SPFeatureMinDeckSize());
        features.add(new SPFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRubles());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureUniqueAristocratsPointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureCardsInHand());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureHandSpaceDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureBuyableCardsInHand());
        features.add(new SPFeatureInteractionTerm(new SPFeatureBuyableCardsInHand(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureDupAristoCount());
        features.add(new SPFeatureInteractionTerm(new SPFeatureDupAristoCount(), new SPFeatureMinDeckSize()));

        // features = new ArrayList<>();
        // // features.add(new SPFeatureMinDeckSize());
        // // features.add(new SPFeaturePoints());
        // // features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureRubles());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureUniqueAristocratsPointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeaturePointsRoundGain());
        // // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureRublesRoundGain());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureCardsInHand());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureCardsInHandDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureHandSpaceDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureBuyableCardsInHand());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureBuyableCardsInHand(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureDupAristoCount());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureDupAristoCount(), new SPFeatureMinDeckSize()));

        initializeModel();
    }

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
            model = (LogisticRegression.Binomial) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public String getCSVHeader() {
        StringBuilder header = new StringBuilder();
        for (SPFeature feature : features) {
            header.append(feature.getName()).append(",");
        }
        header.append("is_winner");
        return header.toString();
    }

    public String getCSVRow(SPState state, boolean[] isWinner) {
        int currentPlayerIndex = state.playerTurn;
        int winnerVal = isWinner[currentPlayerIndex] ? 1 : 0;
        StringBuilder row = new StringBuilder();
        for (SPFeature feature : features) {
            row.append(feature.getValue(state)).append(",");
        }
        row.append(winnerVal);
        return row.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        boolean[] isWinner = transcript.getWinners();
        for (SPState state : transcript.getStates()) {
            rows.append(getCSVRow(state, isWinner)).append("\n");
        }
        return rows.toString();
    }

    public void generateCSVData(String filename, int numGames) {
        int numThreads = 10;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        CompletionService<String> completionService = new ExecutorCompletionService<>(executor);
        
        AtomicInteger gamesCompleted = new AtomicInteger(0);
    
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
    
            for (int i = 0; i < numGames; i++) {
                completionService.submit(() -> {
                    // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
                    SPGameTranscript transcript = SPSimulateGame.simulateGame(new AIDanSPPlayerFMCTrainer(), new AIDanSPPlayerFMCTrainer());
                    return getCSVRows(transcript); 
                });
            }
    
            for (int i = 0; i < numGames; i++) {
                try {
                    Future<String> future = completionService.take(); 
                    
                    String csvRow = future.get(); 
                    
                    writer.print(csvRow);
    
                    int done = gamesCompleted.incrementAndGet();
                    System.out.println("Generated game " + done + "/" + numGames);
    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); 
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    System.err.println("Error during game simulation: " + e.getCause());
                    e.printStackTrace();
                }
            }
    
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown(); 
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow(); 
                Thread.currentThread().interrupt();
            }
        }
    }

    public ArrayList<String> getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        for (SPFeature feature : features) {
            names.add(feature.getName());
        }
        return names;
    }

    public void learnModel() {
        String trainingDataFile = "AIDanSPTrainingDataFMCTrainervsFMCTrainer.csv";
        int numGames = 1000; // Number of games to simulate for training data
        generateCSVData(trainingDataFile, numGames);

        ArrayList<String> desiredFeatureNames = getFeatureNames();
        List<Integer> featureIndicesToKeep = new ArrayList<>();

        List<double[]> values = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            
            String headerLine = br.readLine(); 
            String[] allCsvHeaders = headerLine.split(",");

            for (int i = 0; i < allCsvHeaders.length; i++) {
                String csvColumnName = allCsvHeaders[i];
                if (desiredFeatureNames.contains(csvColumnName)) {
                    featureIndicesToKeep.add(i);
                }
            }
            
            if (featureIndicesToKeep.size() != desiredFeatureNames.size()) {
                System.err.println("ERROR: Mismatch between features in code and CSV!");
                System.err.println("Code features: " + desiredFeatureNames);
                System.err.println("Found " + featureIndicesToKeep.size() + " matching columns in CSV.");
                return;
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                
                // Build the feature row *selectively* using our index map
                double[] row = new double[featureIndicesToKeep.size()];
                for (int i = 0; i < featureIndicesToKeep.size(); i++) {
                    int csvIndex = featureIndicesToKeep.get(i);
                    row[i] = Double.parseDouble(parts[csvIndex]);
                }
                values.add(row);
                
                // Add the label (it's always the last part)
                labels.add(Integer.parseInt(parts[parts.length - 1]));
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][] X = values.toArray(new double[0][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();

        // Perform logistic regression using the Smile library
        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);

        // Save the model to a file using an ObjectOutputStream
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
            oos.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Print the model coefficients along with their feature names
        // This printout will now work correctly!
        System.out.println("Model coefficients:");
        System.out.println(features.size() + " features"); // Will print 14
        System.out.println(model.coefficients().length + " coefficients"); // Will print 15 (14 + intercept)
        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
        for (int i = 0; i < model.coefficients().length - 1; i++) { // Will loop 14 times
            // This is now safe and won't crash
            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
        }
    }

    public double predict(SPState state) {
        // Create a double array for the feature values
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        // Use the logistic regression model to predict the probability of winning
        return model.score(featureValues);
    }

    // min_deck_size – the number of cards in the smallest phase deck
    class SPFeatureMinDeckSize extends SPFeature {
        public SPFeatureMinDeckSize() {
            super("min_deck_size", "the number of cards in the smallest phase deck");
        }

        public Object getValue(SPState state) {
            int minDeckSize = Integer.MAX_VALUE;
            minDeckSize = Math.min(minDeckSize, state.workerDeck.size());
            minDeckSize = Math.min(minDeckSize, state.buildingDeck.size());
            minDeckSize = Math.min(minDeckSize, state.aristocratDeck.size());
            minDeckSize = Math.min(minDeckSize, state.tradingDeck.size());
            return minDeckSize;
        }
    }
    
    // round – current round number
    class SPFeatureRound extends SPFeature {
        public SPFeatureRound() {
            super("round", "current round number");
        }

        public Object getValue(SPState state) {
            return state.round;
        }
    }

    // points – current player points
    class SPFeaturePoints extends SPFeature {
        public SPFeaturePoints() {
            super("points", "current player points");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn];
        }
    }

    // points_diff – current player points relative to the opponent (assumes two players)
    class SPFeaturePointsDiff extends SPFeature {
        public SPFeaturePointsDiff() {
            super("points_diff", "current player points relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
        }
    }

    // rubles – current player rubles (money)
    class SPFeatureRubles extends SPFeature {
        public SPFeatureRubles() {
            super("rubles", "current player rubles (money)");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn];
        }
    }

    // rubles_diff – current player rubles (money) relative to the opponent
    class SPFeatureRublesDiff extends SPFeature {
        public SPFeatureRublesDiff() {
            super("rubles_diff", "current player rubles (money) relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
        }
    }

    // points_round_gain – the number of points the current player is gaining per round
    class SPFeaturePointsRoundGain extends SPFeature {
        public SPFeaturePointsRoundGain() {
            super("points_round_gain", "the number of points the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound;
        }
    }

    // points_round_gain_diff – the number of points the current player is gaining per round relative to the opponent
    class SPFeaturePointsRoundGainDiff extends SPFeature {
        public SPFeaturePointsRoundGainDiff() {
            super("points_round_gain_diff", "the number of points the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound - opponentPointsPerRound;
        }
    }

    // rubles_round_gain – the number of rubles the current player is gaining per round
    class SPFeatureRublesRoundGain extends SPFeature {
        public SPFeatureRublesRoundGain() {
            super("rubles_round_gain", "the number of rubles the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound;
        }
    }

    // rubles_round_gain_diff – the number of rubles the current player is gaining per round relative to the opponent
    class SPFeatureRublesRoundGainDiff extends SPFeature {
        public SPFeatureRublesRoundGainDiff() {
            super("rubles_round_gain_diff", "the number of rubles the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound - opponentRublesPerRound;
        }
    }

    // unique_aristocrats – the number of unique aristocrats of the current player
    class SPFeatureUniqueAristocrats extends SPFeature {
        public SPFeatureUniqueAristocrats() {
            super("unique_aristocrats", "the number of unique aristocrats of the current player");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // unique_aristocrats_diff – the number of unique aristocrats of the current player relative to the opponent
    class SPFeatureUniqueAristocratsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }
    
    // unique_aristocrats_points_diff – the difference in potential point added based on the number of unique aristocrats of the current player relative to the opponent
    class SPFeatureUniqueAristocratsPointsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsPointsDiff() {
            super("unique_aristocrats_points_diff", "the difference in potential point added based on the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS, state.playerAristocrats.get(state.playerTurn).stream().distinct().count());
            long opponentUniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS, state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count());
            return SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) uniqueAristocrats) - SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) opponentUniqueAristocrats);
        }
    }

    // cards_in_hand – the number of cards in the current player hand
    class SPFeatureCardsInHand extends SPFeature {
        public SPFeatureCardsInHand() {
            super("cards_in_hand", "the number of cards in the current player hand");
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(state.playerTurn).size();
        }
    }

    // cards_in_hand_diff – the number of cards in the current player hand relative to the opponent
    class SPFeatureCardsInHandDiff extends SPFeature {
        public SPFeatureCardsInHandDiff() {
            super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent");
        }

        public Object getValue(SPState state) {
            int cardsInHand = state.playerHands.get(state.playerTurn).size();
            int opponentCardsInHand = state.playerHands.get(1 - state.playerTurn).size();
            return cardsInHand - opponentCardsInHand;
        }   
    }
    
    // hand_space_avl – the number of empty slots in a player's hand
    class SPFeatureHandSpace extends SPFeature {
        public SPFeatureHandSpace() {
            super("hand_space_avl", "the number of empty slots in a player's hand");
        }

        public Object getValue(SPState state) {
        	boolean hasWarehouse = state.playerBuildings.get(state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		return (hasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
        }
    }
    
    // hand_space_avl_diff – the difference in number of empty slots in a player's hand
    class SPFeatureHandSpaceDiff extends SPFeature {
        public SPFeatureHandSpaceDiff() {
            super("hand_space_avl_diff", "the number of empty slots in a player's hand");
        }

        public Object getValue(SPState state) {
        	boolean curHasWarehouse = state.playerBuildings.get(state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		int curHandSpace = (curHasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
    		boolean oppHasWarehouse = state.playerBuildings.get(1 - state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		int oppHandSpace = (oppHasWarehouse ? 4 : 3) - state.playerHands.get(1 - state.playerTurn).size();
    		return curHandSpace - oppHandSpace;
        }
    }
    
    // buying_adv - the advantage the current player has over the number of new card he's able to acquired the following round
    class SPFeatureBuyingAdv extends SPFeature {
    	public SPFeatureBuyingAdv() {
    		super("buying_adv", "the advantage the current player has over the number of new card he's able to acquired the following round");
    	}
    	
    	public Object getValue(SPState state) {
    		int numCardsInRows = state.upperCardRow.size() + state.lowerCardRow.size();
    		int nextStartingPlayer = state.startingPlayer[(state.phase + 1) % SPState.NUM_DECKS];
    		if (numCardsInRows % 2 == 0) {
    			return 0;
    		} else {
    			if (nextStartingPlayer == 0) {
    				return 1;
    			} else {
    				return -1;
    			}
    		}
    	}
    }
    
    // total_cost_in_hand - the sum of the costs of all cards in the player's hand. a high number suggests the hand is difficult to empty.
    class SPFeatureCostInHand extends SPFeature {
    	public SPFeatureCostInHand() {
    		super("total_cost_in_hand", "the sum of the costs of all cards in the player's hand. a high number suggests the hand is difficult to empty.");
    	}
    	
    	public Object getValue(SPState state) {
    		int sum = 0;
    		
    		for (SPCard card : state.playerHands.get(state.playerTurn)) {
    			sum += card.cost;
    		}
    		
    		return sum;
    	}
    }
    
    // buyable_cards_in_hand - the number of cards in hand that the player can currently afford to play
    class SPFeatureBuyableCardsInHand extends SPFeature {
    	public SPFeatureBuyableCardsInHand() {
    		super("buyable_cards_in_hand", "the number of cards in hand that the player can currently afford to play");
    	}
    	
    	public Object getValue(SPState state) {
    		int count = 0;
    		
    		for (SPCard card : state.playerHands.get(state.playerTurn)) {
    			if (state.playerRubles[state.playerTurn] > card.cost) {
    				count += 1;
    			}
    		}
    		
    		return count;
    	}
    }
    
    // duplicate_aristocrat_count - the number of Aristocrat cards that are duplicates of others in play. these are prime candidates for upgrades.
    class SPFeatureDupAristoCount extends SPFeature {
    	public SPFeatureDupAristoCount() {
    		super("duplicate_aristocrat_count", "the number of Aristocrat cards that are duplicates of others in play. these are prime candidates for upgrades.");
    	}
    	
    	public Object getValue(SPState state) {
    		return state.playerAristocrats.get(state.playerTurn).size() - state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
    	}
    }

    public static void main(String[] args) {
        new AIDanSPStateFeaturesLR32();
    }

}