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
 * SPCardAnalysisTwo - Balancing Edition (Edition 2 Analysis)
 * - Class Renamed from SPTournamentParallel
 * - Pivot Round: 5
 * - Tracks "Card Type" in the CSV report
 * - Fixes "Pick Rate > 1" by tracking Carryover Appearances
 * - Fixes Observatory tracking
 */
public class SPCardAnalysisTwo {

    private static final int NUM_GAMES = 700; 
    private static final long MILLISECONDS_PER_GAME = 210000;
    private static final int NUM_PROCESSORS = 15;
    
    // --- PIVOT ROUND SET TO 5 ---
    private static final int PIVOT_ROUND = 5; 

    public static final Map<String, StatTracker> globalCardStats = new ConcurrentHashMap<>();

    public static class StatTracker {
        String name;
        String type; // NEW: Tracks Card Type (Worker, Building, etc.)
        
        // Appearance Splits
        int appearancesTotal = 0; // Pure New Arrivals (for Global Pick Rate)
        int appearancesEarly = 0; // Arrivals < 5
        int appearancesLate = 0;  // Arrivals >= 5 + Carryover from R4
        
        // Buy/Win Splits
        int buysTotal = 0;
        int winsTotal = 0;
        int buysEarly = 0;
        int winsEarly = 0;
        int buysLate = 0;
        int winsLate = 0;

        public StatTracker(String name, String type) { 
            this.name = name; 
            this.type = type;
        }
        
        public synchronized void addEarlyAppearance(int count) {
            this.appearancesTotal += count;
            this.appearancesEarly += count;
        }

        public synchronized void addLateAppearance(int count, boolean isCarryover) {
            // If it's carryover, it adds to the "Denominator" of Late stats
            // but DOES NOT add to the "Total Unique Cards Seen" (appearancesTotal)
            this.appearancesLate += count;
            if (!isCarryover) {
                this.appearancesTotal += count;
            }
        }
        
        public synchronized void recordBuy(boolean isWinner, int round) {
            this.buysTotal++;
            if (isWinner) this.winsTotal++;

            if (round < PIVOT_ROUND) {
                this.buysEarly++;
                if (isWinner) this.winsEarly++;
            } else {
                this.buysLate++;
                if (isWinner) this.winsLate++;
            }
        }
    }

    private static class PurchaseRecord {
        String cardName;
        int round;
        public PurchaseRecord(String c, int r) { cardName = c; round = r; }
    }

    private static class SingleGameResult {
        int[] scores = new int[4]; 
        String logOutput;
        
        Map<String, Integer> earlyNewAppearances = new HashMap<>();
        Map<String, Integer> lateNewAppearances = new HashMap<>();
        Map<String, Integer> lateCarryoverAppearances = new HashMap<>(); // Inventory at start of R5
        
        List<PurchaseRecord> p1Purchases = new ArrayList<>();
        List<PurchaseRecord> p2Purchases = new ArrayList<>();
    }

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

                SPState node = new SPState();
                
                // Market tracking state
                Map<String, Integer> previousMarketCounts = new HashMap<>();
                boolean carryoverRecorded = false;

                int move;
                String winner = "DRAW";
                
                while (!node.isGameOver()) {
                    // --- DATA COLLECTION ---
                    
                    // 1. Build Current Market Snapshot
                    Map<String, Integer> currentMarketCounts = new HashMap<>();
                    for (SPCard c : node.upperCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);
                    for (SPCard c : node.lowerCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);
                    
                    // Include Observed Card in Market Counts
                    if (node.observedCard != null) {
                        currentMarketCounts.merge(node.observedCard.name, 1, Integer::sum);
                    }

                    // 2. Handle Carryover (Transition to Late Game)
                    // If we just hit Round 5, everything currently on the board counts as a "Late Appearance"
                    if (node.round >= PIVOT_ROUND && !carryoverRecorded) {
                        for (Map.Entry<String, Integer> entry : previousMarketCounts.entrySet()) {
                            result.lateCarryoverAppearances.merge(entry.getKey(), entry.getValue(), Integer::sum);
                        }
                        carryoverRecorded = true;
                    }

                    // 3. Detect NEW Appearances (Deck Reveals)
                    for (Map.Entry<String, Integer> entry : currentMarketCounts.entrySet()) {
                        String cardName = entry.getKey();
                        int currentCount = entry.getValue();
                        int prevCount = previousMarketCounts.getOrDefault(cardName, 0);

                        if (currentCount > prevCount) {
                            int newInstances = currentCount - prevCount;
                            if (node.round < PIVOT_ROUND) {
                                result.earlyNewAppearances.merge(cardName, newInstances, Integer::sum);
                            } else {
                                result.lateNewAppearances.merge(cardName, newInstances, Integer::sum);
                            }
                        }
                    }
                    previousMarketCounts = currentMarketCounts;
                    // -----------------------

                    int currentPlayer = node.playerTurn;
                    clock.reset();
                    clock.start();
                    move = players[node.playerTurn].getAction((SPState) node.clone(), playerMillisRemaining[node.playerTurn]);
                    clock.stop();

                    try {
                        ArrayList<SPAction> legalActions = node.getLegalActions();
                        if (move < 0 || move >= legalActions.size()) throw new IllegalArgumentException("Invalid Move");
                        SPAction action = legalActions.get(move);
                        
                        // Track Buys
                        if (action instanceof SPBuyAction) {
                            SPBuyAction buy = (SPBuyAction) action;
                            PurchaseRecord record = new PurchaseRecord(buy.card.name, node.round);
                            if (currentPlayer == 0) result.p1Purchases.add(record);
                            else result.p2Purchases.add(record);
                        }
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
                
                if (winner.equals("PLAYER 1 WINS")) result.scores[0]++;
                if (winner.equals("PLAYER 2 WINS")) result.scores[1]++;
                
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
        
        // Initialize Global Stats with Type from SPCard
        for (SPCard c : SPCard.ALL_CARDS) {
            // NEW: Passing c.type to constructor
            globalCardStats.putIfAbsent(c.name, new StatTracker(c.name, c.type));
        }

        String[] playerIdentifiers = {"AIDanSPPlayerFMCTrainerTimeManaged", "AiDanExpectiminimaxPlayerTimeManaged"};
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
                
                // 1. Update Appearances
                for (Map.Entry<String, Integer> e : result.earlyNewAppearances.entrySet()) {
                    if(globalCardStats.containsKey(e.getKey())) 
                        globalCardStats.get(e.getKey()).addEarlyAppearance(e.getValue());
                }
                for (Map.Entry<String, Integer> e : result.lateNewAppearances.entrySet()) {
                    if(globalCardStats.containsKey(e.getKey())) 
                        globalCardStats.get(e.getKey()).addLateAppearance(e.getValue(), false); // New arrival
                }
                for (Map.Entry<String, Integer> e : result.lateCarryoverAppearances.entrySet()) {
                    if(globalCardStats.containsKey(e.getKey())) 
                        globalCardStats.get(e.getKey()).addLateAppearance(e.getValue(), true); // Carryover
                }
                
                // 2. Update Buys
                boolean p1Won = (result.scores[0] > 0); 
                boolean p2Won = (result.scores[1] > 0);

                for (PurchaseRecord pr : result.p1Purchases) {
                    if (globalCardStats.containsKey(pr.cardName)) {
                        globalCardStats.get(pr.cardName).recordBuy(p1Won, pr.round);
                    }
                }
                for (PurchaseRecord pr : result.p2Purchases) {
                    if (globalCardStats.containsKey(pr.cardName)) {
                        globalCardStats.get(pr.cardName).recordBuy(p2Won, pr.round);
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        return matchScores;
    }

    private static void generateBalanceReport() throws Exception {
        File reportFile = new File("CardBalanceReport_Edition4.csv");
        try (PrintWriter out = new PrintWriter(reportFile)) {
            // Header updated with "Type" column
            out.println("Card Name,Type,Total Appearances,Total Buys,Pick Rate,Win Rate," +
                        "Early Apps,Early Buys,Early Pick Rate,Early Win Rate," + 
                        "Late Apps (inc Carryover),Late Buys,Late Pick Rate,Late Win Rate,Status");

            List<String> sortedKeys = globalCardStats.keySet().stream().sorted().collect(Collectors.toList());

            for (String key : sortedKeys) {
                StatTracker stats = globalCardStats.get(key);
                
                // Total Metrics (Based on NEW arrivals only)
                double pickRate = (stats.appearancesTotal > 0) ? (double) stats.buysTotal / stats.appearancesTotal : 0.0;
                double winRate = (stats.buysTotal > 0) ? (double) stats.winsTotal / stats.buysTotal : 0.0;

                // Early Metrics
                double earlyPickRate = (stats.appearancesEarly > 0) ? (double) stats.buysEarly / stats.appearancesEarly : 0.0;
                double earlyWinRate = (stats.buysEarly > 0) ? (double) stats.winsEarly / stats.buysEarly : 0.0;

                // Late Metrics (Denominator includes Carryover to prevent > 1.0)
                double latePickRate = (stats.appearancesLate > 0) ? (double) stats.buysLate / stats.appearancesLate : 0.0;
                double lateWinRate = (stats.buysLate > 0) ? (double) stats.winsLate / stats.buysLate : 0.0;

                String status = "Balanced";
                if (pickRate > 0.90) status = "MUST BUY";
                else if (pickRate < 0.15) status = "IGNORE";
                else if (winRate > 0.60 && stats.buysTotal > 50) status = "WIN CONDITION";
                else if (earlyWinRate > 0.70) status = "EARLY GAME SNOWBALL";
                else if (lateWinRate > 0.70) status = "LATE GAME CLOSER";

                // Added stats.type to output
                out.printf("\"%s\",\"%s\",%d,%d,%.2f,%.2f,%d,%d,%.2f,%.2f,%d,%d,%.2f,%.2f,\"%s\"%n", 
                    key, stats.type,
                    stats.appearancesTotal, stats.buysTotal, pickRate, winRate,
                    stats.appearancesEarly, stats.buysEarly, earlyPickRate, earlyWinRate,
                    stats.appearancesLate, stats.buysLate, latePickRate, lateWinRate,
                    status);
            }
        }
        System.out.println("Balance report written to: " + reportFile.getAbsolutePath());
    }
}