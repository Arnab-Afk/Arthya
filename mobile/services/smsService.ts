/**
 * SMS Reader Service
 * Handles reading SMS messages from the device (Android only)
 */

import { Platform, PermissionsAndroid, Alert, Linking } from 'react-native';
import { SMSMessage, ParsedTransaction, parseMultipleSMS, isTransactionSMS } from './smsParser';

// We'll use expo-modules or a native bridge for SMS reading
// For now, this provides the interface and Android implementation

export interface SMSReadResult {
  success: boolean;
  messages: SMSMessage[];
  error?: string;
}

export interface SMSPermissionResult {
  granted: boolean;
  error?: string;
}

/**
 * Check if SMS reading is supported on this platform
 */
export function isSMSReadingSupported(): boolean {
  return Platform.OS === 'android';
}

/**
 * Request SMS read permission (Android only)
 */
export async function requestSMSPermission(): Promise<SMSPermissionResult> {
  if (Platform.OS !== 'android') {
    return {
      granted: false,
      error: 'SMS reading is only supported on Android',
    };
  }

  try {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.READ_SMS,
      {
        title: 'SMS Permission',
        message: 'Arthya needs access to your SMS messages to automatically track your transactions from bank messages.',
        buttonNeutral: 'Ask Me Later',
        buttonNegative: 'Cancel',
        buttonPositive: 'OK',
      }
    );

    return {
      granted: granted === PermissionsAndroid.RESULTS.GRANTED,
      error: granted !== PermissionsAndroid.RESULTS.GRANTED 
        ? 'SMS permission denied' 
        : undefined,
    };
  } catch (error: any) {
    return {
      granted: false,
      error: error.message || 'Failed to request SMS permission',
    };
  }
}

/**
 * Check if SMS permission is already granted
 */
export async function checkSMSPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') {
    return false;
  }

  try {
    const result = await PermissionsAndroid.check(
      PermissionsAndroid.PERMISSIONS.READ_SMS
    );
    return result;
  } catch {
    return false;
  }
}

/**
 * Open app settings for manual permission grant
 */
export function openAppSettings(): void {
  Linking.openSettings();
}

/**
 * Show permission denied alert with option to open settings
 */
export function showPermissionDeniedAlert(): void {
  Alert.alert(
    'Permission Required',
    'SMS permission is required to automatically import transactions. Please enable it in app settings.',
    [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Open Settings', onPress: openAppSettings },
    ]
  );
}

// Native module interface for reading SMS
// This will be implemented via react-native-get-sms-android or a custom native module
interface NativeSMSModule {
  list(
    filter: string,
    error: (err: string) => void,
    success: (count: number, smsList: string) => void
  ): void;
}

// Try to get the native module dynamically
let SmsAndroid: NativeSMSModule | null = null;

// Initialize on first use
const loadSmsModule = async (): Promise<void> => {
  if (SmsAndroid !== null) return;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const module = require('react-native-get-sms-android');
    SmsAndroid = module.default || module;
  } catch {
    console.log('SMS native module not available - using mock data');
  }
};

/**
 * Read SMS messages from the device
 */
export async function readSMS(options: {
  maxCount?: number;
  minDate?: Date;
  box?: 'inbox' | 'sent' | 'draft';
}): Promise<SMSReadResult> {
  const { maxCount = 500, minDate, box = 'inbox' } = options;

  if (Platform.OS !== 'android') {
    return {
      success: false,
      messages: [],
      error: 'SMS reading is only supported on Android',
    };
  }

  // Check permission first
  const hasPermission = await checkSMSPermission();
  if (!hasPermission) {
    const permResult = await requestSMSPermission();
    if (!permResult.granted) {
      return {
        success: false,
        messages: [],
        error: permResult.error,
      };
    }
  }

  // Try to load native module
  await loadSmsModule();

  // If native module is not available, return mock data
  if (!SmsAndroid) {
    console.log('SMS module not available - using mock data for development');
    return {
      success: true,
      messages: getMockSMSMessages(),
    };
  }

  return new Promise((resolve) => {
    const filter: Record<string, any> = {
      box,
      maxCount,
    };

    if (minDate) {
      filter.minDate = minDate.getTime();
    }

    SmsAndroid!.list(
      JSON.stringify(filter),
      (error: string) => {
        resolve({
          success: false,
          messages: [],
          error,
        });
      },
      (_count: number, smsList: string) => {
        try {
          const messages: SMSMessage[] = JSON.parse(smsList);
          resolve({
            success: true,
            messages,
          });
        } catch {
          resolve({
            success: false,
            messages: [],
            error: 'Failed to parse SMS messages',
          });
        }
      }
    );
  });
}

/**
 * Read and parse transaction SMS messages
 */
export async function getTransactionsFromSMS(options?: {
  maxCount?: number;
  daysBack?: number;
}): Promise<{
  success: boolean;
  transactions: ParsedTransaction[];
  totalSMS: number;
  transactionSMS: number;
  error?: string;
}> {
  const { maxCount = 500, daysBack = 30 } = options || {};

  const minDate = new Date();
  minDate.setDate(minDate.getDate() - daysBack);

  const result = await readSMS({ maxCount, minDate });

  if (!result.success) {
    return {
      success: false,
      transactions: [],
      totalSMS: 0,
      transactionSMS: 0,
      error: result.error,
    };
  }

  const transactionMessages = result.messages.filter(isTransactionSMS);
  const transactions = parseMultipleSMS(transactionMessages);

  return {
    success: true,
    transactions,
    totalSMS: result.messages.length,
    transactionSMS: transactionMessages.length,
  };
}

/**
 * Mock SMS messages for development/testing
 */
function getMockSMSMessages(): SMSMessage[] {
  const now = Date.now();
  const day = 24 * 60 * 60 * 1000;

  return [
    {
      _id: '1',
      address: 'HDFCBK',
      body: 'Rs.2,500.00 debited from A/c XX1234 on 28-11-24. UPI Ref: 433445566778. Available balance: Rs.45,000.50. To Swiggy.',
      date: now - day,
      read: 1,
      type: 1,
    },
    {
      _id: '2',
      address: 'SBIINB',
      body: 'Your A/c X1234 is credited with Rs.50,000.00 on 27-11-24. Salary for Nov 2024. Avl Bal Rs.95,000.50',
      date: now - 2 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '3',
      address: 'ICICIB',
      body: 'Dear Customer, Rs.1,200 has been debited from your account XX5678 for Amazon Pay. Ref No: 998877665544. Balance: Rs.23,400',
      date: now - 3 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '4',
      address: 'AXISBK',
      body: 'INR 500.00 spent on your Axis Bank Debit Card XX9012 at UBER INDIA. Avl Lmt: INR 49,500.00',
      date: now - 4 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '5',
      address: 'PAYTMB',
      body: 'Rs. 150 paid to Zomato from Paytm Wallet. Txn ID: PTM123456789. New balance: Rs.350',
      date: now - 5 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '6',
      address: 'HDFCBK',
      body: 'Rs.3,000.00 credited to A/c XX1234. Refund from Flipkart. Ref: REF123456. Balance: Rs.48,000.50',
      date: now - 6 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '7',
      address: 'KOTAKB',
      body: 'Your a/c XX4567 is debited for Rs.899 towards Netflix subscription. Available Balance Rs.12,100',
      date: now - 7 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '8',
      address: 'GPAY',
      body: 'Paid Rs.250 to BigBasket via UPI. UPI Ref: 778899001122. From A/c ending 1234',
      date: now - 8 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '9',
      address: 'SBIINB',
      body: 'ATM WDL of Rs.5,000 from A/c XX1234 at SBI ATM on 20-11-24. Avl Bal Rs.40,000',
      date: now - 9 * day,
      read: 1,
      type: 1,
    },
    {
      _id: '10',
      address: 'HDFCBK',
      body: 'Rs.15,000 transferred to A/c XX9999. IMPS Ref: IMPS456789. From XX1234. Bal: Rs.25,000',
      date: now - 10 * day,
      read: 1,
      type: 1,
    },
  ];
}
