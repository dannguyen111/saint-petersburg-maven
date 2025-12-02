// Assumed structure of your CardStats.java
public class CardStats {
    private String name;
    private int quantity;
    private String type;
    private int cost;
    private int rubles;
    private int appearances = 0;
    private int buys = 0;
    private int wins = 0;

    public CardStats(String name) { 
        this.name = name;
    }

    public synchronized void addAppearance() { this.appearances++; }
    public synchronized void addBuy() { this.buys++; }
    public synchronized void addWin() { this.wins++; }

    public int getAppearances() { return appearances; }
    public int getBuys() { return buys; }
    public int getWins() { return wins; }
}