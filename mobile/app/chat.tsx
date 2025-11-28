import React, { useState, useRef, useEffect } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  FlatList, 
  KeyboardAvoidingView, 
  Platform,
  ActivityIndicator 
} from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInUp } from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import api from '@/services/api';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  suggestions?: string[];
}

export default function ChatScreen() {
  const router = useRouter();
  const flatListRef = useRef<FlatList>(null);
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: '1', 
      text: 'Hello! I\'m Arthya AI, your personal financial coach. I can help you with:\n\n💰 Savings tips\n📊 Budget advice\n🎯 Goal tracking\n📈 Spending analysis\n💡 Investment basics\n\nWhat would you like to know?', 
      sender: 'ai',
      suggestions: ['How can I save more?', 'Analyze my spending', 'Show my goals']
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<{ role: 'user' | 'model'; content: string }[]>([]);

  const sendMessage = async (text?: string) => {
    const messageText = text || inputText.trim();
    if (!messageText || isLoading) return;

    const userMsg: Message = { 
      id: Date.now().toString(), 
      text: messageText, 
      sender: 'user' 
    };
    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);

    // Update conversation history
    const newHistory = [...conversationHistory, { role: 'user' as const, content: messageText }];
    setConversationHistory(newHistory);

    try {
      const response = await api.chatWithCoach(messageText, conversationHistory);
      
      if (response.success && response.data) {
        const aiMsg: Message = {
          id: (Date.now() + 1).toString(),
          text: response.data.message,
          sender: 'ai',
          suggestions: response.data.suggestions || []
        };
        setMessages(prev => [...prev, aiMsg]);
        
        // Update history with AI response
        setConversationHistory([
          ...newHistory, 
          { role: 'model' as const, content: response.data.message }
        ]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I couldn\'t process your request. Please try again.',
        sender: 'ai'
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionPress = (suggestion: string) => {
    sendMessage(suggestion);
  };

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    if (flatListRef.current && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  const renderItem = ({ item, index }: { item: Message; index: number }) => (
    <Animated.View
      entering={FadeInUp.delay(index * 50).duration(300)}
      style={[
        styles.messageBubble,
        item.sender === 'user' ? styles.userBubble : styles.aiBubble
      ]}
    >
      {item.sender === 'ai' && (
        <View style={styles.aiHeader}>
          <View style={styles.aiIcon}>
            <Ionicons name="sparkles" size={14} color={Colors.primary} />
          </View>
          <Text style={styles.aiLabel}>Arthya AI</Text>
        </View>
      )}
      <Text style={[
        styles.messageText,
        item.sender === 'user' ? styles.userText : styles.aiText
      ]}>{item.text}</Text>
      
      {item.suggestions && item.suggestions.length > 0 && (
        <View style={styles.suggestionsContainer}>
          {item.suggestions.map((suggestion, idx) => (
            <TouchableOpacity 
              key={idx} 
              style={styles.suggestionChip}
              onPress={() => handleSuggestionPress(suggestion)}
            >
              <Text style={styles.suggestionText}>{suggestion}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </Animated.View>
  );

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={Colors.text} />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <View style={styles.headerIconContainer}>
            <Ionicons name="sparkles" size={20} color={Colors.primary} />
          </View>
          <View>
            <Text style={styles.headerTitle}>Arthya AI</Text>
            <Text style={styles.headerSubtitle}>Powered by Gemini</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.menuButton}>
          <Ionicons name="ellipsis-vertical" size={20} color={Colors.text} />
        </TouchableOpacity>
      </View>

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      />

      {/* Loading indicator */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <View style={styles.loadingBubble}>
            <ActivityIndicator size="small" color={Colors.primary} />
            <Text style={styles.loadingText}>Thinking...</Text>
          </View>
        </View>
      )}

      {/* Input */}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Ask about your finances..."
          placeholderTextColor={Colors.textDim}
          multiline
          maxLength={500}
          editable={!isLoading}
          onSubmitEditing={() => sendMessage()}
        />
        <TouchableOpacity 
          onPress={() => sendMessage()} 
          style={[styles.sendButton, isLoading && styles.sendButtonDisabled]}
          disabled={isLoading || !inputText.trim()}
        >
          <Ionicons name="send" size={20} color={isLoading ? Colors.textDim : "#000"} />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
    backgroundColor: Colors.background,
  },
  backButton: {
    padding: 8,
  },
  headerCenter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: `${Colors.primary}22`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: '600',
  },
  headerSubtitle: {
    color: Colors.textDim,
    fontSize: 12,
  },
  menuButton: {
    padding: 8,
  },
  listContent: {
    padding: 16,
    paddingBottom: 8,
  },
  messageBubble: {
    maxWidth: '85%',
    padding: 14,
    borderRadius: 20,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: Colors.primary,
    borderBottomRightRadius: 6,
  },
  aiBubble: {
    alignSelf: 'flex-start',
    backgroundColor: Colors.card,
    borderBottomLeftRadius: 6,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  aiHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 6,
  },
  aiIcon: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: `${Colors.primary}22`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  aiLabel: {
    color: Colors.primary,
    fontSize: 12,
    fontWeight: '600',
  },
  messageText: {
    fontSize: 15,
    lineHeight: 22,
  },
  userText: {
    color: '#000',
  },
  aiText: {
    color: Colors.text,
  },
  suggestionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
    gap: 8,
  },
  suggestionChip: {
    backgroundColor: `${Colors.primary}22`,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.primary,
  },
  suggestionText: {
    color: Colors.primary,
    fontSize: 13,
    fontWeight: '500',
  },
  loadingContainer: {
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  loadingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: Colors.card,
    padding: 12,
    borderRadius: 16,
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  loadingText: {
    color: Colors.textDim,
    fontSize: 14,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 32 : 16,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    alignItems: 'flex-end',
    backgroundColor: Colors.background,
  },
  input: {
    flex: 1,
    backgroundColor: Colors.card,
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 12,
    color: Colors.text,
    marginRight: 12,
    borderWidth: 1,
    borderColor: Colors.border,
    fontSize: 16,
    maxHeight: 100,
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 5,
  },
  sendButtonDisabled: {
    backgroundColor: Colors.card,
    shadowOpacity: 0,
    elevation: 0,
  },
});
