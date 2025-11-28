import React from 'react';
import { StyleSheet, View, Text, ScrollView, Image, TouchableOpacity, Dimensions, Platform } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SummaryCard } from '@/components/SummaryCard';
import { GoalCard } from '@/components/GoalCard';
import Animated, { FadeInDown, FadeInRight, ZoomIn } from 'react-native-reanimated';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      {/* Header */}
      {/* Header */}
      <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
        <TouchableOpacity>
          <Ionicons name="menu" size={24} color={Colors.text} />
        </TouchableOpacity>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.iconButton}>
            <Ionicons name="search" size={24} color={Colors.text} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.iconButton}>
            <Ionicons name="notifications-outline" size={24} color={Colors.text} />
            <View style={styles.badge} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.profileButton}>
            <Text style={styles.profileText}>N</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Cards Section */}
        {/* Cards Section */}
        <Animated.ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.cardsScroll}
          entering={FadeInRight.delay(200).duration(500)}
        >
          <LinearGradient
            colors={['#333', '#111']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.card}
          >
            <View style={styles.cardHeader}>
              <Ionicons name="logo-paypal" size={24} color="#fff" />
              <Ionicons name="wifi" size={24} color="#666" />
            </View>
            <Text style={styles.cardNumber}>6277  2154  6598  3247</Text>
            <View style={styles.cardFooter}>
              <Text style={styles.cardLabel}>Expir 05/22</Text>
            </View>
          </LinearGradient>

          <LinearGradient
            colors={['#222', '#000']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[styles.card, { marginLeft: 15, opacity: 0.6 }]}
          >
            <View style={styles.cardHeader}>
              <Text style={styles.payeerText}>PAYEER</Text>
            </View>
            <Text style={styles.cardNumber}>6591 4784 2033</Text>
          </LinearGradient>
        </Animated.ScrollView>

        {/* Summary Cards */}
        {/* Summary Cards */}
        <Animated.View entering={FadeInDown.delay(300).duration(500)} style={styles.summarySection}>
          <SummaryCard title="Income" amount="$5,200" icon="arrow-down" color="#9FE8AE" trend="+12%" />
          <SummaryCard title="Expense" amount="$1,450" icon="arrow-up" color="#FF453A" trend="-5%" />
          <SummaryCard title="Savings" amount="$3,750" icon="wallet" color="#D4B483" trend="+8%" />
        </Animated.View>

        {/* Available Amount */}
        {/* Available Amount */}
        <Animated.View entering={FadeInDown.delay(400).duration(500)} style={styles.balanceSection}>
          <Text style={styles.balanceLabel}>Available Amount</Text>
          <Text style={styles.balanceValue}>$24,178.25</Text>

          <View style={styles.balanceRow}>
            <View style={styles.balanceItem}>
              <View style={[styles.indicator, { borderColor: Colors.primary }]} />
              <View>
                <Text style={styles.itemLabel}>Card balance</Text>
                <Text style={styles.itemValue}>$24,178.25</Text>
              </View>
            </View>
            <View style={styles.balanceItem}>
              <View style={[styles.indicator, { borderColor: Colors.accent }]} />
              <View>
                <Text style={styles.itemLabel}>Credit limit</Text>
                <Text style={styles.itemValue}>$50,000</Text>
              </View>
            </View>
          </View>
        </Animated.View>

        {/* Goals & Milestones */}
        {/* Goals & Milestones */}
        <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.goalsSection}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Goals & Milestones</Text>
            <TouchableOpacity>
              <Text style={styles.seeAllText}>See All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.goalsScroll}>
            <GoalCard
              title="New Car"
              currentAmount="$15,000"
              targetAmount="$25,000"
              progress={0.6}
              icon="car-sport"
            />
            <GoalCard
              title="Holiday"
              currentAmount="$2,500"
              targetAmount="$5,000"
              progress={0.5}
              icon="airplane"
            />
            <GoalCard
              title="MacBook"
              currentAmount="$1,200"
              targetAmount="$2,000"
              progress={0.6}
              icon="laptop-outline"
            />
          </ScrollView>
        </Animated.View>

        {/* Quick Transfer */}
        {/* Quick Transfer */}
        <Animated.View entering={FadeInDown.delay(600).duration(500)} style={styles.transferSection}>
          <Text style={styles.sectionTitle}>Quick Transfer</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.transferScroll}>
            {[
              { name: 'Alex', id: '3281', img: 'https://i.pravatar.cc/100?img=11' },
              { name: 'Hanna', id: '5714', img: 'https://i.pravatar.cc/100?img=5' },
              { name: 'Emma', id: '2357', img: 'https://i.pravatar.cc/100?img=9' },
              { name: 'John', id: '5785', img: 'https://i.pravatar.cc/100?img=13' },
            ].map((user, index) => (
              <View key={index} style={styles.userCard}>
                <Image source={{ uri: user.img }} style={styles.userImage} />
                <Text style={styles.userName}>{user.name}</Text>
                <Text style={styles.userId}>**** {user.id}</Text>
              </View>
            ))}
          </ScrollView>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    paddingTop: 60,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconButton: {
    marginLeft: 20,
  },
  badge: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.accent,
    borderWidth: 1,
    borderColor: Colors.background,
  },
  profileButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.card,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 20,
    borderWidth: 2,
    borderColor: Colors.primary,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  profileText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  content: {
    paddingBottom: 20,
  },
  cardsScroll: {
    paddingLeft: 24,
    marginBottom: 32,
  },
  card: {
    width: width * 0.75,
    height: 190,
    borderRadius: 24,
    padding: 24,
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    marginRight: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  payeerText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 20,
    letterSpacing: 1,
  },
  cardNumber: {
    color: '#E0E0E0',
    fontSize: 20,
    letterSpacing: 3,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    marginTop: 20,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  cardLabel: {
    color: '#AAAAAA',
    fontSize: 12,
  },
  balanceSection: {
    marginBottom: 32,
    padding: 20,
    backgroundColor: Colors.card,
    marginHorizontal: 24,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  balanceLabel: {
    color: Colors.textDim,
    fontSize: 16,
    marginBottom: 8,
  },
  balanceValue: {
    color: Colors.text,
    fontSize: 40,
    fontWeight: 'bold',
    marginBottom: 24,
  },
  balanceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  balanceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '48%',
  },
  indicator: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 4,
    marginRight: 12,
  },
  itemLabel: {
    color: Colors.textDim,
    fontSize: 13,
  },
  itemValue: {
    color: Colors.text,
    fontWeight: '600',
    fontSize: 16,
    marginTop: 4,
  },
  transferSection: {
    paddingHorizontal: 24,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },
  transferScroll: {
    marginLeft: -8,
  },
  userCard: {
    backgroundColor: Colors.card,
    width: 80,
    height: 120,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: Colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.12,
    shadowRadius: 5,
    elevation: 3,
  },
  userImage: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginBottom: 12,
  },
  userName: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: '500',
  },
  userId: {
    color: Colors.textDim,
    fontSize: 11,
    marginTop: 4,
  },
  summarySection: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  goalsSection: {
    marginBottom: 32,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  seeAllText: {
    color: Colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  goalsScroll: {
    paddingLeft: 24,
  },
});
