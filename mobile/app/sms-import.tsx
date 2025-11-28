/**
 * SMS Import Screen
 * Full screen for importing transactions from SMS messages
 */

import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { SMSImportModal } from '@/components/SMSImport';
import { Colors } from '@/constants/Colors';

export default function SMSImportScreen() {
  const router = useRouter();

  const handleClose = () => {
    router.back();
  };

  const handleImportComplete = (count: number) => {
    console.log(`Imported ${count} transactions`);
    // Optionally navigate back or refresh data
  };

  return (
    <SafeAreaView style={styles.container}>
      <SMSImportModal 
        onClose={handleClose} 
        onImportComplete={handleImportComplete} 
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
});
