import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * SPTournamentParallel - Balancing Edition (Frequency Tracking)
 * Modified to track Total Card Appearances, Buys, and Win Rates.
 */
public class SPCardAnalysisOne {

    private static final int NUM_GAMES = 550; 
    private static final long MILLISECONDS_PER_GAME = 120000;
    private static final int NUM_PROCESSORS = 11; 

    // Global Registry for Card Stats (Thread-Safe)
    public static final Map<String, CardStats> globalCardStats = new ConcurrentHashMap<>();

    /**
     * Helper class to store the result of a single game.
     */
    private static class SingleGameResult {
        int[] scores = new int[4]; 
        String logOutput;
        
        // Map to track how many times each card appeared in this specific game
        // Key: Card Name, Value: Total Count of new appearances
        Map<String, Integer> appearanceCounts = new HashMap<>();
        
        List<String> winnerBoughtCards = new ArrayList<>(); 
        List<String> allBoughtCards = new ArrayList<>(); 
    }

    /**
     * A callable task that runs *one single game*.
     */
    private static class SingleGameTask implements Callable<SingleGameResult> {
        private final String player1Class;
        private final String player2Class;
        private final int gameNumber;

        SingleGameTask(String player1Class, String player2Class, int gameNumber) {
            this.player1Class = player1Class;
            this.player2Class = player2Class;
            this.gameNumber = gameNumber;
        }

        @Override
        public SingleGameResult call() {
            SingleGameResult result = new SingleGameResult();
            StringWriter sw = new StringWriter();
            PrintWriter p = new PrintWriter(sw);

            try {
                SPPlayer[] players = new SPPlayer[2];
                Class<?> playerClass = Class.forName(player1Class);
                players[0] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();
                playerClass = Class.forName(player2Class);
                players[1] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();

                long[] playerMillisRemaining = {MILLISECONDS_PER_GAME/2L, MILLISECONDS_PER_GAME/2L};
                Stopwatch clock = new Stopwatch();
                long timeTaken;

                SPState node = new SPState();
                
                List<String> p1Buys = new ArrayList<>();
                List<String> p2Buys = new ArrayList<>();

                // Tracking Market State for Frequency Calculation
                Map<String, Integer> previousMarketCounts = new HashMap<>();

                int move;
                String winner = "DRAW";
                
                while (!node.isGameOver()) {
                    // --- DATA COLLECTION: Track NEW Appearances ---
                    // 1. Build count of currently visible cards
                    Map<String, Integer> currentMarketCounts = new HashMap<>();
                    for (SPCard c : node.upperCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);
                    for (SPCard c : node.lowerCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);
                    
                    // --- BUG FIX START: Include cards held in Observatory ---
                    if (node.observedCard != null) {
                        currentMarketCounts.merge(node.observedCard.name, 1, Integer::sum);
                    }
                    // --- BUG FIX END ---

                    // 2. Compare with previous state to find new entries
                    for (Map.Entry<String, Integer> entry : currentMarketCounts.entrySet()) {
                        String cardName = entry.getKey();
                        int currentCount = entry.getValue();
                        int prevCount = previousMarketCounts.getOrDefault(cardName, 0);

                        // If we see more of card X than before, new copies must have been dealt
                        if (currentCount > prevCount) {
                            int newAppearances = currentCount - prevCount;
                            result.appearanceCounts.merge(cardName, newAppearances, Integer::sum);
                        }
                    }
                    // 3. Update snapshot for next turn
                    previousMarketCounts = currentMarketCounts;
                    // ------------------------------------------

                    int currentPlayer = node.playerTurn;

                    clock.reset();
                    clock.start();
                    move = players[node.playerTurn].getAction((SPState) node.clone(), playerMillisRemaining[node.playerTurn]);
                    timeTaken = clock.stop();

                    playerMillisRemaining[node.playerTurn] -= timeTaken;
                    if (playerMillisRemaining[node.playerTurn] < 0) {
                        winner = (node.playerTurn == 0) ? "PLAYER 2 WINS" : "PLAYER 1 WINS";
                        break;
                    }

                    try {
                        ArrayList<SPAction> legalActions = node.getLegalActions();
                        if (move < 0 || move >= legalActions.size()) {
                            throw new IllegalArgumentException("Invalid Action Index");
                        }
                        SPAction action = legalActions.get(move);
                        
                        // --- DATA COLLECTION: Track Buys ---
                        if (action instanceof SPBuyAction) {
                            SPBuyAction buy = (SPBuyAction) action;
                            String cardName = buy.card.name;
                            result.allBoughtCards.add(cardName);
                            if (currentPlayer == 0) p1Buys.add(cardName);
                            else p2Buys.add(cardName);
                        }
                        // -----------------------------------

                        action.take();
                    } catch (Exception e) {
                        winner = (currentPlayer == 0) ? "PLAYER 2 WINS" : "PLAYER 1 WINS";
                        break;
                    }
                }

                if (node.isGameOver()) {
                    if (node.playerPoints[0] > node.playerPoints[1]) winner = "PLAYER 1 WINS";
                    else if (node.playerPoints[0] < node.playerPoints[1]) winner = "PLAYER 2 WINS";
                    else winner = "DRAW";
                }

                if (winner.equals("PLAYER 1 WINS")) {
                    result.scores[0]++;
                    result.winnerBoughtCards.addAll(p1Buys);
                } else if (winner.equals("PLAYER 2 WINS")) {
                    result.scores[1]++;
                    result.winnerBoughtCards.addAll(p2Buys);
                }
                
                result.scores[2] += node.playerPoints[0];
                result.scores[3] += node.playerPoints[1];
                System.out.println("Game " + (gameNumber + 1) + " finished.");

            } catch (Exception e) {
                e.printStackTrace(p);
            } finally {
                result.logOutput = sw.toString();
            }
            return result;
        }
    }

    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        
        // Initialize Global Stats
        for (SPCard c : SPCard.ALL_CARDS) {
            globalCardStats.putIfAbsent(c.name, new CardStats(c.name));
        }

        // AIs for testing
        String[] playerIdentifiers = {"AIDanSPPlayerFMCTrainerTimeManaged", "AIDanSPPlayerFMCTrainerTimeManaged", "SPPlayerFlatMC"};
        String[] competitors = playerIdentifiers;
        
        File logFolder = new File("logs");
        logFolder.mkdir();
        
        ExecutorService executor = Executors.newFixedThreadPool(NUM_PROCESSORS);

        for (int i = 0; i < competitors.length; i++) {
            for (int j = 0; j < competitors.length; j++) {
                if (i >= j) continue;
                System.out.println("Starting match: " + competitors[i] + " vs. " + competitors[j]);
                runGames(competitors[i], competitors[j], new PrintWriter(System.out), executor);
            }
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        generateBalanceReport();
        System.out.println("Analysis Complete. Time: " + (System.currentTimeMillis() - startTime) + "ms");
    }

    private static int[] runGames(String player1, String player2, PrintWriter p, ExecutorService executor) {
        int[] matchScores = new int[4];
        List<Future<SingleGameResult>> gameFutures = new ArrayList<>();

        for (int g = 0; g < NUM_GAMES; g++) {
            gameFutures.add(executor.submit(new SingleGameTask(player1, player2, g)));
        }

        for (Future<SingleGameResult> future : gameFutures) {
            try {
                SingleGameResult result = future.get();
                
                // --- AGGREGATE STATS ---
                // 1. Update Total Appearances (Looping to add count)
                for (Map.Entry<String, Integer> entry : result.appearanceCounts.entrySet()) {
                    String name = entry.getKey();
                    int count = entry.getValue();
                    if (globalCardStats.containsKey(name)) {
                        // Add the specific number of appearances found in this game
                        for(int k=0; k<count; k++) {
                            globalCardStats.get(name).addAppearance();
                        }
                    }
                }
                
                // 2. Update Buys
                for (String cardName : result.allBoughtCards) {
                    if (globalCardStats.containsKey(cardName)) {
                        globalCardStats.get(cardName).addBuy();
                    }
                }
                
                // 3. Update Wins
                for (String cardName : result.winnerBoughtCards) {
                    if (globalCardStats.containsKey(cardName)) {
                        globalCardStats.get(cardName).addWin();
                    }
                }
                // -----------------------

                matchScores[0] += result.scores[0];
                matchScores[1] += result.scores[1];
                matchScores[2] += result.scores[2];
                matchScores[3] += result.scores[3];

            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        return matchScores;
    }

    private static void generateBalanceReport() throws Exception {
        File reportFile = new File("CardBalanceReport_Edition1.csv");
        try (PrintWriter out = new PrintWriter(reportFile)) {
            // Header updated to reflect total frequency
            out.println("Card Name,Total Appearances,Total Buys,Wins,Pick Rate (Buys/Apps),Win Rate,Status");

            List<String> sortedKeys = globalCardStats.keySet().stream().sorted().collect(Collectors.toList());

            for (String key : sortedKeys) {
                CardStats stats = globalCardStats.get(key);
                
                int apps = stats.getAppearances(); 
                int buys = stats.getBuys();
                int wins = stats.getWins();
                
                // Pick Rate is now "Conversion Rate"
                double pickRate = (apps > 0) ? (double) buys / apps : 0.0;
                double winRate = (buys > 0) ? (double) wins / buys : 0.0;

                String status = "Balanced";
                if (pickRate > 0.90) status = "MUST BUY (High Priority)";
                else if (pickRate < 0.15) status = "IGNORE (Weak)";
                else if (winRate > 0.60 && buys > 50) status = "WIN CONDITION";

                out.printf("\"%s\",%d,%d,%d,%.2f,%.2f,\"%s\"%n", 
                    key, apps, buys, wins, pickRate, winRate, status);
            }
        }
        System.out.println("Balance report written to: " + reportFile.getAbsolutePath());
    }
}