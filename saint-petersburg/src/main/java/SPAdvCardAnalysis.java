import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * SPAdvCardAnalysis - Advanced Balancing Edition
 * Tracks:
 * 1. Frequency/Pick Rate
 * 2. Early (Round < 4) vs Late (Round >= 4) Win Rates
 * 3. Cost Efficiency (ROI)
 */
public class SPAdvCardAnalysis {

    private static final int NUM_GAMES = 11; // Higher count for stable stats
    private static final long MILLISECONDS_PER_GAME = 300000; 
    private static final int NUM_PROCESSORS = 11; 
    private static final int PIVOT_ROUND = 4; // Definition of "Late Game"

    // Thread-safe global stats registry
    public static final Map<String, StatTracker> globalStats = new ConcurrentHashMap<>();

    /**
     * Advanced Statistics Container for a specific Card
     */
    public static class StatTracker {
        String name;
        int appearances = 0;
        
        // Overall
        int totalBuys = 0;
        int totalWins = 0;

        // Temporal Stats (Early < Round 4 <= Late)
        int buysEarly = 0;
        int winsEarly = 0;
        int buysLate = 0;
        int winsLate = 0;

        // Efficiency Stats
        // Accumulator for ROI calculation
        double totalRubEfficiencySum = 0.0;
        double totalPointsEfficiencySum = 0.0;
        int efficiencyDataPoints = 0;

        public StatTracker(String name) { this.name = name; }

        public synchronized void addAppearance(int count) { this.appearances += count; }
        
        public synchronized void recordPurchase(boolean wonGame, int round, double roirub, double roipoints) {
            totalBuys++;
            if (wonGame) totalWins++;

            // Temporal Split
            if (round < PIVOT_ROUND) {
                buysEarly++;
                if (wonGame) winsEarly++;
            } else {
                buysLate++;
                if (wonGame) winsLate++;
            }

            // Efficiency
            totalRubEfficiencySum += roirub;
            totalPointsEfficiencySum += roipoints;
            efficiencyDataPoints++;
        }
    }

    /**
     * Records a specific buy event during a game to be processed after game end.
     */
    private static class PurchaseRecord {
        String cardName;
        int roundBought;
        int costPaid;
        int playerIndex;
        int baseIncome;
        int rublesIncome;
        int pointsIncome; // Rubles + Points per turn

        PurchaseRecord(String name, int round, int cost, int player, int income, int rubles, int points) {
            this.cardName = name;
            this.roundBought = round;
            this.costPaid = cost;
            this.playerIndex = player;
            this.baseIncome = income;
            this.rublesIncome = rubles;
            this.pointsIncome = points;
        }
    }

    private static class SingleGameResult {
        int[] scores = new int[4]; 
        String logOutput;
        Map<String, Integer> appearanceCounts = new HashMap<>();
        List<PurchaseRecord> purchases = new ArrayList<>();
        int winnerIndex = -1;
        int finalRound = 0;
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
                Map<String, Integer> previousMarketCounts = new HashMap<>();

                while (!node.isGameOver()) {
                    // --- 1. Track Appearances ---
                    Map<String, Integer> currentMarketCounts = new HashMap<>();
                    for (SPCard c : node.upperCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);
                    for (SPCard c : node.lowerCardRow) currentMarketCounts.merge(c.name, 1, Integer::sum);

                    for (Map.Entry<String, Integer> entry : currentMarketCounts.entrySet()) {
                        String name = entry.getKey();
                        int current = entry.getValue();
                        int prev = previousMarketCounts.getOrDefault(name, 0);
                        if (current > prev) {
                            result.appearanceCounts.merge(name, current - prev, Integer::sum);
                        }
                    }
                    previousMarketCounts = currentMarketCounts;
                    // ----------------------------

                    int currentPlayer = node.playerTurn;
                    clock.reset();
                    clock.start();
                    int move = players[node.playerTurn].getAction((SPState) node.clone(), playerMillisRemaining[node.playerTurn]);
                    long timeTaken = clock.stop();
                    playerMillisRemaining[node.playerTurn] -= timeTaken;

                    if (playerMillisRemaining[node.playerTurn] < 0) {
                        result.winnerIndex = (node.playerTurn == 0) ? 1 : 0; // Opponent wins
                        break;
                    }

                    try {
                        ArrayList<SPAction> legalActions = node.getLegalActions();
                        SPAction action = legalActions.get(move);
                        
                        // --- 2. Track Buys with Context ---
                        if (action instanceof SPBuyAction) {
                            SPBuyAction buy = (SPBuyAction) action;
                            // Capture "Income" as Points + Rubles defined on the card
                            // Note: This is static potential. 
                            int income = buy.card.points + buy.card.rubles;
                            
                            result.purchases.add(new PurchaseRecord(
                                buy.card.name,
                                node.round,
                                buy.cost, // The actual cost paid (includes discounts)
                                currentPlayer,
                                income,
                                buy.card.rubles,
                                buy.card.points
                            ));
                        }
                        // ----------------------------------
                        
                        action.take();
                    } catch (Exception e) {
                        result.winnerIndex = (currentPlayer == 0) ? 1 : 0;
                        break;
                    }
                }

                if (result.winnerIndex == -1 && node.isGameOver()) {
                    if (node.playerPoints[0] > node.playerPoints[1]) result.winnerIndex = 0;
                    else if (node.playerPoints[0] < node.playerPoints[1]) result.winnerIndex = 1;
                    // Draw handling: winnerIndex remains -1
                }
                
                result.finalRound = node.round;
                
                // Update Basic Scores
                if (result.winnerIndex != -1) result.scores[result.winnerIndex]++;
                result.scores[2] += node.playerPoints[0];
                result.scores[3] += node.playerPoints[1];

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
        
        // Pre-fill global stats
        for (SPCard c : SPCard.ALL_CARDS) {
            globalStats.putIfAbsent(c.name, new StatTracker(c.name));
        }

        String[] competitors = {"AIDanSPPlayerFMCTrainerTimeManaged", "AIDanSPPlayerFMCTrainerTimeManaged"};
        ExecutorService executor = Executors.newFixedThreadPool(NUM_PROCESSORS);

        for (int i = 0; i < competitors.length; i++) {
            for (int j = 0; j < competitors.length; j++) {
                if (i >= j) continue;
                System.out.println("Running match: " + competitors[i] + " vs " + competitors[j]);
                runGames(competitors[i], competitors[j], new PrintWriter(System.out), executor);
            }
        }

        executor.shutdown();
        executor.awaitTermination(2, TimeUnit.HOURS);

        generateBalanceReport();
        System.out.println("Done. Duration: " + (System.currentTimeMillis() - startTime) + "ms");
    }

    private static void runGames(String p1, String p2, PrintWriter p, ExecutorService executor) {
        List<Future<SingleGameResult>> futures = new ArrayList<>();
        for (int g = 0; g < NUM_GAMES; g++) {
            futures.add(executor.submit(new SingleGameTask(p1, p2, g)));
        }

        for (Future<SingleGameResult> f : futures) {
            try {
                SingleGameResult res = f.get();
                
                // 1. Aggregate Appearances
                res.appearanceCounts.forEach((name, count) -> {
                    if (globalStats.containsKey(name)) globalStats.get(name).addAppearance(count);
                });

                // 2. Calculate Metrics for Purchases
                for (PurchaseRecord pr : res.purchases) {
                    if (!globalStats.containsKey(pr.cardName)) continue;

                    boolean isWinner = (pr.playerIndex == res.winnerIndex);
                    
                    // Formula: (Points + Rubles) * RoundsRemaining / Cost
                    // RoundsRemaining Estimate: FinalRound - BoughtRound + 1 (inclusive of current)
                    int roundsActive = Math.max(1, res.finalRound - pr.roundBought + 1);
                    double totalRubGenerated = pr.rublesIncome * roundsActive;
                    double totalPointsGenerated = pr.pointsIncome * roundsActive;
                    
                    // Avoid division by zero if cost was discounted to 0 (e.g. Carpenter)
                    // Treat cost 0 as cost 0.5 to avoid Infinity, or cap ROI. 
                    double effectiveCost = Math.max(1.0, pr.costPaid);
                    double rubEff = totalRubGenerated / effectiveCost;
                    double pointsEff = totalPointsGenerated / effectiveCost;

                    globalStats.get(pr.cardName).recordPurchase(isWinner, pr.roundBought, rubEff, pointsEff);
                }

            } catch (Exception e) { e.printStackTrace(); }
        }
    }

    private static void generateBalanceReport() throws Exception {
        File report = new File("CardBalanceReport_Advanced.csv");
        try (PrintWriter out = new PrintWriter(report)) {
            out.println("Card Name,Appearances,Pick Rate,Overall Win Rate," +
                        "Early Win Rate (R1-3),Late Win Rate (R4+),Avg Efficiency (ROI),Avg Rubles Efficiency (ROI),Avg Points Efficiency (ROI),Status");

            List<String> keys = globalStats.keySet().stream().sorted().collect(Collectors.toList());

            for (String key : keys) {
                StatTracker st = globalStats.get(key);
                
                double pickRate = (st.appearances > 0) ? (double) st.totalBuys / st.appearances : 0.0;
                double winRate = (st.totalBuys > 0) ? (double) st.totalWins / st.totalBuys : 0.0;
                
                // Early vs Late Win Rates
                double earlyWinRate = (st.buysEarly > 0) ? (double) st.winsEarly / st.buysEarly : 0.0;
                double lateWinRate = (st.buysLate > 0) ? (double) st.winsLate / st.buysLate : 0.0;
                
                // Avg Efficiency
                double rubEff = (st.efficiencyDataPoints > 0) 
                    ? st.totalRubEfficiencySum / st.efficiencyDataPoints 
                    : 0.0;

                double pointsEff = (st.efficiencyDataPoints > 0) 
                    ? st.totalPointsEfficiencySum / st.efficiencyDataPoints 
                    : 0.0;

                double avgEff = rubEff + pointsEff;

                // Basic automated tagging
                String status = "Balanced";
                if (pickRate > 0.9) status = "AUTO-PICK";
                else if (pickRate < 0.1) status = "TRASH";
                else if (earlyWinRate > 0.65) status = "SNOWBALL CARD";
                else if (avgEff > 3.0) status = "HIGH EFFICIENCY";

                out.printf("\"%s\",%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,\"%s\"%n", 
                    key, st.appearances, pickRate, winRate, earlyWinRate, lateWinRate, avgEff, rubEff, pointsEff, status);
            }
        }
        System.out.println("Report generated: " + report.getAbsolutePath());
    }
}