import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter; // Added for game-level logging
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit; // Added for error handling

/**
 * Tournament.java - Simple SaintPetersburg tournament software
 * See the TODO comments below.  This software assumes that all player class files are located in the same directory and have a unique player ID as a prefix
 * to their SPPlayer subclasses.  List these unique player classnames in the String[] playerIdentifiers.
 * The software will create detailed player logs of games in a "log/" subdirectory, naming files according to the 0-based index of 
 * player numbers in playerIdentifiers.
 * A SPResults.csv file will be generated summarizing tournament results.
 * While running, the console will report high-level progress in the round-robin tournament.
 * If a round-robin tournament is not desired (e.g. too many players), please develop alternative tournament code and share it!
 *
 * This version is modified to run all *games* within a single *match* in parallel,
 * while processing the *matches* sequentially.
 * * @author Todd W. Neller (Original)
 * * On our system: java -cp "/usr/local/smile/lib/*:." SPTournament > tournament.log &
 * */
public class SPTournament {

	private static final int NUM_GAMES = 11; // TODO - set the number of games per match (e.g., 11)
	private static final long MILLISECONDS_PER_GAME = 60000; // TODO - set the maximum milliseconds per game
	// (300000 ms/game = 300 seconds/game = 5 minutes/game)
    
    // TODO - set the number of processors (threads) to use for parallel games
    private static final int NUM_PROCESSORS = 10; 

    /**
     * Helper class to store the result of a single game.
     * This is returned by SingleGameTask.
     */
    private static class SingleGameResult {
        // [p1_win, p2_win, p1_points, p2_points]
        int[] scores = new int[4]; 
        String logOutput; // Log output for just this game
    }

    /**
     * A callable task that runs *one single game* between two players.
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
            // Each game task writes to its own StringWriter to avoid garbled logs
            StringWriter sw = new StringWriter();
            PrintWriter p = new PrintWriter(sw);

            try {
                SPPlayer[] players = new SPPlayer[2];
                Class<?> playerClass = Class.forName(player1Class);
                players[0] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();
                playerClass = Class.forName(player2Class);
                players[1] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();

                p.println("Game " + (gameNumber + 1) + " started. Player 1: " + player1Class + ", Player 2: " + player2Class);
                long[] playerMillisRemaining = {MILLISECONDS_PER_GAME/2L, MILLISECONDS_PER_GAME/2L};

                // Create a clock
                Stopwatch clock = new Stopwatch();
                long timeTaken;

                // Create a node with a random board initial state
                SPState node = new SPState();
                p.println(node);

                // While game is on...
                int move;
                String winner = "DRAW";
                while (!node.isGameOver()) {
                    int currentPlayer = node.playerTurn;

                    // Request move from current player
                    clock.reset();
                    clock.start();
                    move = players[node.playerTurn].getAction((SPState) node.clone(), playerMillisRemaining[node.playerTurn]);
                    timeTaken = clock.stop();

                    // Deduct time taken
                    playerMillisRemaining[node.playerTurn] -= timeTaken;
                    if (playerMillisRemaining[node.playerTurn] < 0) {
                        if (node.playerTurn == 0) {
                            p.println("Player 1 game timer expired.");
							node.playerPoints[1] += 50; // TODO: Adjust timeout penalty/bonus
							winner = "PLAYER 2 WINS";
                        } else {
                            p.println("Player 2 game timer expired.");
                            node.playerPoints[0] += 50; // TODO: Adjust timeout penalty/bonus
							winner = "PLAYER 1 WINS";
                        }
                        break;
                    }

                    try {
                        // Update game state
                        ArrayList<SPAction> legalActions = node.getLegalActions();
                        if (move < 0 || move >= legalActions.size()) {
                            throw new IllegalArgumentException("Player " + ((currentPlayer == 0) ? "1" : "2") + " chose an invalid action index: " + move);
                        }
                        SPAction action = legalActions.get(move);
                        p.println("> " + action);
                        p.println();
                        action.take();
                        
                        // Display Progress
                        p.println(node);
                    }
                    catch (Exception e) {
                        p.println("ERROR: Player " + ((currentPlayer == 0) ? "1" : "2") + " makes an ILLEGAL MOVE and forfeits the game.");
                        p.println(e.getMessage());
                        e.printStackTrace(p);
                        winner = (currentPlayer == 0) ? "PLAYER 2 WINS" : "PLAYER 1 WINS";
                        if (currentPlayer == 0) node.playerPoints[1] += 50; // TODO: Adjust forfeit penalty/bonus
                        else node.playerPoints[0] += 50; // TODO: Adjust forfeit penalty/bonus
                        break;
                    }
                }

                // Display winner and statistics
                if (node.isGameOver()) {
                    if (node.playerPoints[0] > node.playerPoints[1])
                        winner = "PLAYER 1 WINS";
                    else
                        if (node.playerPoints[0] < node.playerPoints[1])
                            winner = "PLAYER 2 WINS";
                        else
                            winner = "DRAW";
                }
                p.println("Time Taken (ms): ");
                long t = MILLISECONDS_PER_GAME / 2L - playerMillisRemaining[0];
                p.println("Player 1: " + t);
                t = MILLISECONDS_PER_GAME / 2L - playerMillisRemaining[1];
                p.println("Player 2: " + t);
                p.println(winner);

                if (winner.equals("PLAYER 1 WINS"))
                    result.scores[0]++; // p1 win
                if (winner.equals("PLAYER 2 WINS"))
                    result.scores[1]++; // p2 win
                
                result.scores[2] += node.playerPoints[0]; // p1 points
                result.scores[3] += node.playerPoints[1]; // p2 points
                
                p.println("-----------------------------------------------------------------------------------");

            } catch (Exception e) {
                p.println("CRITICAL ERROR IN SingleGameTask for " + player1Class + " vs " + player2Class);
                e.printStackTrace(p);
            } finally {
                p.flush();
                result.logOutput = sw.toString();
            }
            return result;
        }
    }


	/**
	 * Run a round-robin SaintPetersburg tournament with given hard-coded parameters noted with "TODO" comments.
	 * @param args (unused)
	 * @throws Exception An exception is raised if there is a difficulty writing out results to files.
	 */
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();

		// TODO - List class names for the competing SPPlayer class file names.
//		String[] playerIdentifiers = {"SPPlayerFlatMC", "AIDanSPPlayerHybridObjective", "JFMTPlayer", "MaiNePlayer", "MiMaPlayer", "OKHybridPlayer"};
//		String[] playerIdentifiers = {"SPPlayerFlatMC", "AIDanSPPlayerHybridObjective", "MaiNePlayer"};
		// String[] playerIdentifiers = {"SPPlayerFlatMC", "AiDanExpectiminimaxPlayer", "JFMTPlayer2",  "MiMaPlayer", "MaiNeNNPlayer", "OKTurnBasedFeaturesPlayer"};
		//String[] playerIdentifiers = {"AiDanExpectiminimaxPlayer", "OKTurnBasedFeaturesPlayer"};
		String[] playerIdentifiers = {"AIDanSPPlayerHybridObjective", "AIDanSPPlayerFMCTrainer"};

		String[] competitors = new String[playerIdentifiers.length];
		for (int i = 0; i < playerIdentifiers.length; i++)
			competitors[i] = playerIdentifiers[i] ;
		File logFolder = new File("logs");
		logFolder.mkdir();
		File results = new File("SPTournamentResults.csv");
		
		int[] wins = new int[competitors.length]; // Match wins
		int[] totalWins = new int[wins.length]; // Game wins
		int[] points = new int[wins.length]; // Total points
        
        // Storage for CSV results
        String[][] csvMatchResults = new String[competitors.length][competitors.length];

        // --- Create ONE thread pool to be reused for all matches ---
        ExecutorService executor = Executors.newFixedThreadPool(NUM_PROCESSORS);
        System.out.println("Created thread pool with " + NUM_PROCESSORS + " threads.");
        System.out.println("Running " + NUM_GAMES + " games per match.");

		for (int i = 0; i < competitors.length; i++) {
			for (int j = 0; j < competitors.length; j++) {
				if (i >= j) { // Don't play a player against itself or against an opponent already played.
                    csvMatchResults[i][j] = ""; // Fill placeholder for CSV
					continue;
				}

                // --- Run one match sequentially ---
				String logFilename =  "logs/tournament." + i + "." + j + ".log";    
				PrintWriter logOutput = new PrintWriter(new File(logFilename)); // Create log file for play details.
				
                System.out.println("Starting match: " + playerIdentifiers[i] + " (i=" + i + ") vs. " + playerIdentifiers[j] + " (j=" + j + ")");
                
                // This function now runs all games in parallel and waits for them
                int[] scores = runGames(competitors[i], competitors[j], logOutput, executor);
                
                logOutput.flush();
				logOutput.close();
                System.out.println("Finished match: " + playerIdentifiers[i] + " (i=" + i + ") vs. " + playerIdentifiers[j] + " (j=" + j + ")");

                // Aggregate stats
                // scores[] = [p1_game_wins, p2_game_wins, p1_total_points, p2_total_points]
				if (scores[0] > scores[1])
					wins[i]++; // p1 match win
				if (scores[1] > scores[0])
					wins[j]++; // p2 match win
				totalWins[i] += scores[0]; // p1 game wins
				totalWins[j] += scores[1]; // p2 game wins
				points[i] += scores[2]; // p1 points
				points[j] += scores[3]; // p2 points

                // Store CSV string
                csvMatchResults[i][j] = "\"(" + scores[0] + ":" + scores[1] + ")\"";
			}
		}

        // --- All matches are done, shut down the executor ---
        System.out.println("All matches completed. Shutting down thread pool...");
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                System.err.println("Thread pool did not shut down gracefully. Forcing shutdown.");
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // --- Write Results to CSV ---
        System.out.println("Writing results to " + results.getName());
		PrintWriter resultsOut = new PrintWriter(results); // output to CSV results file

		resultsOut.println("\"TOURNAMENT RESULTS:\"");
		resultsOut.println("\"First Player (Win:Loss) Ratios:\"");
		resultsOut.println(",\"Second Player\"");
		resultsOut.print("\"First Player\"");
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.print(",\"" + i + "-" + playerIdentifiers[i] + "\"");
		}
		resultsOut.println();
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.print("\"" + i + "-" + playerIdentifiers[i] + "\"");
			for (int j = 0; j < competitors.length; j++) {
				resultsOut.print(",");
                if (csvMatchResults[i][j] != null) {
				    resultsOut.print(csvMatchResults[i][j]);
                }
			}
			resultsOut.println();
		}
		resultsOut.println();
		resultsOut.println();
		resultsOut.println("\"SUMMARY STATISTICS\"");
		resultsOut.println("\"The winner of this table is determined by most winning matches.\"");
		resultsOut.println("\"Ties are broken according to total game wins.\"");
		resultsOut.println("\"Further ties are broken according to total points across all games.\"");
		resultsOut.println(",\"Match Wins\",\"Total Game Wins\",\"Total Points\"");
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.println(i + "-" + playerIdentifiers[i] + "," + wins[i] + "," + totalWins[i] + "," + points[i]);
		}
		resultsOut.close();

		System.out.println("Time taken to calculate results: " + (System.currentTimeMillis() - startTime) + "ms");
	}

	/**
	 * Run <code>NUM_GAMES</code> games of SPPlayers in parallel using the provided ExecutorService.
	 * This method blocks until all games in the match are complete.
	 * * @param player1 - class name of first player.
	 * @param player2 - class name of second player.
	 * @param p - PrintWriter for printing the *summary* of the match to a file.
     * @param executor - The thread pool to submit game tasks to.
	 * @return an int array containing the aggregated results for the match:
     * [player 1 games won, player 2 games won, player 1 total points, player 2 total points]
	 */
	private static int[] runGames(String player1, String player2, PrintWriter p, ExecutorService executor) {
        // [p1_game_wins, p2_game_wins, p1_total_points, p2_total_points]
        int[] matchScores = new int[4];
        List<Future<SingleGameResult>> gameFutures = new ArrayList<>();

        // --- Submit all games for this match to the thread pool ---
        for (int g = 0; g < NUM_GAMES; g++) {
            SingleGameTask gameTask = new SingleGameTask(player1, player2, g);
            gameFutures.add(executor.submit(gameTask));
        }

        // --- Wait for all games to complete and aggregate results ---
        for (Future<SingleGameResult> future : gameFutures) {
            try {
                SingleGameResult result = future.get(); // This line blocks until the game is done
                
                // Write the individual game log to the main match log file
                p.print(result.logOutput);

                // Aggregate scores
                matchScores[0] += result.scores[0]; // p1 game wins
                matchScores[1] += result.scores[1]; // p2 game wins
                matchScores[2] += result.scores[2]; // p1 points
                matchScores[3] += result.scores[3]; // p2 points

            } catch (InterruptedException | ExecutionException e) {
                p.println("CRITICAL ERROR: A game task failed to execute.");
                e.printStackTrace(p);
            }
        }

        // --- Write match summary to the log ---
        p.println("-----------------------------------------------------------------------------------");
        p.println("--- MATCH SUMMARY ---");
        p.println("Player 1 (" + player1 + ") Game Wins: " + matchScores[0]);
		p.println("Player 2 (" + player2 + ") Game Wins: " + matchScores[1]);
		p.println("Player 1 Total Points: " + matchScores[2]);
		p.println("Player 2 Total Points: " + matchScores[3]);
        p.println("-----------------------------------------------------------------------------------");
        p.flush();
        
        return matchScores;
	}
} // Tournament