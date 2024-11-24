from ib_insync import *

def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1) # 7496 for real trading, 7497 for paper trading
        print("Successfully connected to IBKR")
        
        # Get account summary
        account_summary = ib.accountSummary()
        for summary in account_summary:
            print(f"{summary.tag}: {summary.value}")
        
    except Exception as e:
        print(f"Failed to connect: {str(e)}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    main()
