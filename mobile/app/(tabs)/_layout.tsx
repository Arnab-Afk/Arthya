import { Tabs, useRouter } from 'expo-router';
import React from 'react';
import { Platform, View, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

import { HapticTab } from '@/components/haptic-tab';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';

export default function TabLayout() {
  const router = useRouter();

  return (
    <View style={{ flex: 1 }}>
      <Tabs
        screenOptions={{
          tabBarActiveTintColor: Colors.primary,
          tabBarInactiveTintColor: Colors.textDim,
          headerShown: false,
          tabBarButton: HapticTab,
          tabBarStyle: {
            backgroundColor: Colors.tabBackground,
            borderTopWidth: 1,
            borderTopColor: Colors.border,
            elevation: 0,
            height: Platform.OS === 'ios' ? 88 : 60,
            paddingBottom: Platform.OS === 'ios' ? 20 : 8,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: -2 },
            shadowOpacity: 0.1,
            shadowRadius: 8,
          },
        }}>
        <Tabs.Screen
          name="charts"
          options={{
            title: 'Charts',
            tabBarIcon: ({ color }) => <Ionicons size={24} name="bar-chart" color={color} />,
          }}
        />
        <Tabs.Screen
          name="index"
          options={{
            title: 'Home',
            tabBarIcon: ({ color }) => <Ionicons size={24} name="home" color={color} />,
          }}
        />
        <Tabs.Screen
          name="analytics"
          options={{
            title: 'Analytics',
            tabBarIcon: ({ color }) => <Ionicons size={24} name="pie-chart" color={color} />,
          }}
        />
        <Tabs.Screen
          name="explore"
          options={{
            title: 'Profile',
            tabBarIcon: ({ color }) => <Ionicons size={24} name="person" color={color} />,
          }}
        />
      </Tabs>

      {/* AI Chat FAB */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => router.push('/chat')}
        activeOpacity={0.8}
      >
        <LinearGradient
          colors={[Colors.primary, '#2A3B30']}
          style={styles.fabGradient}
        >
          <Ionicons name="chatbubble-ellipses" size={24} color="#000" />
        </LinearGradient>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 100 : 80,
    right: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.30,
    shadowRadius: 4.65,
    elevation: 8,
  },
  fabGradient: {
    width: '100%',
    height: '100%',
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
